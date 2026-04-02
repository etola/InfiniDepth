from dataclasses import dataclass
from typing import Literal, Optional
import os

import numpy as np
import torch
import torch.nn.functional as F
import tyro

from InfiniDepth.utils.inference_utils import (
    OUTPUT_RESOLUTION_MODES,
    apply_sky_mask_to_depth,
    build_scaled_intrinsics_matrix,
    prepare_metric_depth_inputs,
    resolve_camera_intrinsics_for_inference,
    resolve_depth_output_paths,
    resolve_output_size_from_mode,
    run_optional_sky_mask,
)
from InfiniDepth.utils.io_utils import (
    depth2pcd,
    depth_to_disparity,
    load_image,
    load_image_from_array,
    plot_depth,
    save_depth_array,
    save_sampled_point_clouds,
)
from InfiniDepth.utils.model_utils import build_model
from InfiniDepth.utils.sampling_utils import SAMPLING_METHODS


@dataclass
class DepthInferenceArgs:
    # Inputs
    input_image_path: str
    input_depth_path: Optional[str] = None

    # Outputs
    depth_output_dir: Optional[str] = None
    pcd_output_dir: Optional[str] = None
    save_pcd: bool = True

    # Model
    model_type: str = "InfiniDepth_DepthSensor"  # [InfiniDepth, InfiniDepth_DepthSensor]
    depth_model_path: str = "checkpoints/depth/infinidepth_depthsensor.ckpt"
    moge2_pretrained: str = "checkpoints/moge-2-vitl-normal/model.pt"  # Metric depth via MoGe-2 (used when input_depth_path is None)

    # Camera intrinsics
    fx_org: Optional[float] = None
    fy_org: Optional[float] = None
    cx_org: Optional[float] = None
    cy_org: Optional[float] = None

    # Data Resolution
    input_size: tuple[int, int] = (768, 1024)
    output_size: tuple[int, int] = (768, 1024)
    output_resolution_mode: Literal["upsample", "original", "specific"] = "upsample"
    upsample_ratio: int = 1

    # Optional sky segmentation
    enable_skyseg_model: bool = False
    sky_model_ckpt_path: str = "checkpoints/sky/skyseg.onnx"


@dataclass
class DepthInferenceResult:
    input_image_path: str
    org_img: torch.Tensor
    image: torch.Tensor
    query_2d_uniform_coord: torch.Tensor
    pred_2d_uniform_depth: torch.Tensor
    pred_depthmap: torch.Tensor
    org_h: int
    org_w: int
    input_h: int
    input_w: int
    output_h: int
    output_w: int
    fx_org: float
    fy_org: float
    cx_org: float
    cy_org: float
    fx: float
    fy: float
    cx: float
    cy: float
    intrinsics_source: str
    depth_scale_align_factor: float = 1.0
    depth_scale_align_valid_pixels: int = 0

    def output_intrinsics_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


def load_depth_model(args: DepthInferenceArgs) -> tuple[torch.nn.Module, torch.device]:
    if args.output_resolution_mode not in OUTPUT_RESOLUTION_MODES:
        raise ValueError(
            f"Unsupported output_resolution_mode: {args.output_resolution_mode}. "
            f"Choose from {OUTPUT_RESOLUTION_MODES}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference in this script.")

    model = build_model(
        args.model_type,
        model_path=args.depth_model_path,
    )
    print(f"Loaded model: {model.__class__.__name__}")
    return model, torch.device("cuda")


def _depth_map_spatial_shape(depth_map: np.ndarray) -> tuple[int, int]:
    d = np.squeeze(np.asarray(depth_map))
    if d.ndim != 2:
        raise ValueError(
            f"depth_map must be 2D or broadcastable to HxW after squeeze, got shape {np.asarray(depth_map).shape}"
        )
    return int(d.shape[0]), int(d.shape[1])


def assert_depth_aspect_matches_image(
    depth_h: int,
    depth_w: int,
    org_h: int,
    org_w: int,
    *,
    rtol: float = 0.02,
    atol: float = 1e-5,
) -> None:
    """Require depth and RGB to share the same aspect ratio before resize-to-input_size."""
    ri = org_h / float(org_w)
    rd = depth_h / float(depth_w)
    if not np.isclose(ri, rd, rtol=rtol, atol=atol):
        raise ValueError(
            f"Depth map size {depth_h}x{depth_w} (aspect {rd:.6f}) does not match image "
            f"{org_h}x{org_w} (aspect {ri:.6f}). Intrinsics are defined for the image pixels; "
            "the depth grid must match that aspect ratio (sizes may differ)."
        )


def load_rgb_for_depth_inference(
    *,
    frame_image_path: Optional[str],
    input_image_rgb: Optional[np.ndarray],
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Load and resize RGB to ``input_size`` (same behavior as ``load_image``), from a path or HxWx3 array.
    If ``input_image_rgb`` is set, it is used for pixels and ``frame_image_path`` is ignored for loading
    (but may still be used elsewhere as a label).
    """
    if input_image_rgb is not None:
        org_img, image, (org_h, org_w) = load_image_from_array(input_image_rgb, input_size)
        return org_img, image, org_h, org_w
    if frame_image_path is None:
        raise ValueError("Provide `input_image_rgb` or a valid `frame_image_path`.")
    org_img, image, (org_h, org_w) = load_image(frame_image_path, input_size)
    return org_img, image, org_h, org_w


def load_and_preprocess_depth_inference_inputs(
    *,
    input_size: tuple[int, int],
    device: torch.device,
    moge2_pretrained: str,
    frame_image_path: Optional[str],
    input_image_rgb: Optional[np.ndarray],
    frame_depth_path: Optional[str],
    input_depth_map: Optional[np.ndarray],
    depth_load_kwargs: Optional[dict],
    skip_metric_depth_inputs: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
    Optional[tuple[float, float, float, float]],
]:
    """
    Resize RGB to model input size, optionally validate depth aspect ratio vs RGB, and build
    metric-depth tensors for the model (MoGe-2 or file/array depth), matching ``prepare_metric_depth_inputs``.

    When ``skip_metric_depth_inputs`` is True (InfiniDepth + external override), only RGB is loaded;
    depth tensors are left None and MoGe-2 is not run.
    """
    org_img, image, org_h, org_w = load_rgb_for_depth_inference(
        frame_image_path=frame_image_path,
        input_image_rgb=input_image_rgb,
        input_size=input_size,
    )
    image = image.to(device)

    if skip_metric_depth_inputs:
        return org_img, image, org_h, org_w, None, None, None, False, None

    if input_depth_map is not None:
        dh, dw = _depth_map_spatial_shape(input_depth_map)
        assert_depth_aspect_matches_image(dh, dw, org_h, org_w)

    gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics = prepare_metric_depth_inputs(
        input_depth_path=frame_depth_path,
        input_size=input_size,
        image=image,
        device=device,
        moge2_pretrained=moge2_pretrained,
        depth_load_kwargs=depth_load_kwargs,
        input_depth_array=input_depth_map,
    )
    return org_img, image, org_h, org_w, gt_depth, prompt_depth, gt_depth_mask, use_gt_depth, moge2_intrinsics


@torch.no_grad()
def run_depth_inference(
    args: DepthInferenceArgs,
    *,
    model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    input_image_path: Optional[str] = None,
    input_depth_path: Optional[str] = None,
    fx_org: Optional[float] = None,
    fy_org: Optional[float] = None,
    cx_org: Optional[float] = None,
    cy_org: Optional[float] = None,
    override_gt_depth=None,
    override_gt_depth_mask=None,
    input_image_rgb: Optional[np.ndarray] = None,
    input_depth_map: Optional[np.ndarray] = None,
    depth_load_kwargs: Optional[dict] = None,
) -> DepthInferenceResult:
    """
    ``fx_org``, ``fy_org``, ``cx_org``, ``cy_org`` refer to the **original** full-resolution image
    (``org_h`` x ``org_w``), matching file-based inference.

    ``input_image_rgb`` (HxWx3) and ``input_depth_map`` (HxW, same aspect ratio as that RGB, any size)
    skip disk I/O; depth is resized with the same pipeline as ``load_depth``. ``depth_load_kwargs``
    is forwarded to that loader (e.g. ``enable_noise_filter``, ``num_samples``).
    """
    if model is None or device is None:
        model, device = load_depth_model(args)

    frame_image_path = input_image_path or args.input_image_path
    frame_depth_path = input_depth_path if input_depth_path is not None else args.input_depth_path

    skip_metric_depth_inputs = args.model_type == "InfiniDepth" and override_gt_depth is not None

    if args.model_type == "InfiniDepth_DepthSensor":
        has_metric_depth = (
            input_depth_map is not None
            or (frame_depth_path is not None and os.path.exists(frame_depth_path))
        )
        assert has_metric_depth, (
            "InfiniDepth_DepthSensor requires a valid input depth map for depth completion. "
            "Provide --input_depth_path or pass `input_depth_map`."
        )

    (
        org_img,
        image,
        org_h,
        org_w,
        gt_depth,
        prompt_depth,
        gt_depth_mask,
        use_gt_depth,
        moge2_intrinsics,
    ) = load_and_preprocess_depth_inference_inputs(
        input_size=args.input_size,
        device=device,
        moge2_pretrained=args.moge2_pretrained,
        frame_image_path=frame_image_path,
        input_image_rgb=input_image_rgb,
        frame_depth_path=frame_depth_path,
        input_depth_map=input_depth_map,
        depth_load_kwargs=depth_load_kwargs,
        skip_metric_depth_inputs=skip_metric_depth_inputs,
    )

    if not skip_metric_depth_inputs:
        if use_gt_depth:
            if input_depth_map is not None:
                print("metric depth from in-memory depth map")
            elif frame_depth_path is not None:
                print(f"metric depth from `{frame_depth_path}`")
        else:
            print(f"metric depth from `{args.moge2_pretrained}`")

    if override_gt_depth is not None:
        gt_depth = _to_single_depth_tensor(
            override_gt_depth,
            device=device,
            dtype=torch.float32,
        )
        if gt_depth.shape[-2:] != image.shape[-2:]:
            gt_depth = F.interpolate(
                gt_depth,
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if override_gt_depth_mask is None:
            gt_depth_mask = torch.isfinite(gt_depth) & (gt_depth > 1e-6)
        else:
            gt_depth_mask = _to_single_depth_tensor(
                override_gt_depth_mask,
                device=device,
                dtype=torch.float32,
            )
            if gt_depth_mask.shape[-2:] != image.shape[-2:]:
                gt_depth_mask = F.interpolate(
                    gt_depth_mask,
                    size=image.shape[-2:],
                    mode="nearest",
                )
            gt_depth_mask = gt_depth_mask > 0.5
            gt_depth_mask &= torch.isfinite(gt_depth) & (gt_depth > 1e-6)

        print("metric depth from external override")

    frame_fx_org = args.fx_org if fx_org is None else fx_org
    frame_fy_org = args.fy_org if fy_org is None else fy_org
    frame_cx_org = args.cx_org if cx_org is None else cx_org
    frame_cy_org = args.cy_org if cy_org is None else cy_org

    frame_fx_org, frame_fy_org, frame_cx_org, frame_cy_org, intrinsics_source = resolve_camera_intrinsics_for_inference(
        fx_org=frame_fx_org,
        fy_org=frame_fy_org,
        cx_org=frame_cx_org,
        cy_org=frame_cy_org,
        org_h=org_h,
        org_w=org_w,
        image=image,
        moge2_pretrained=args.moge2_pretrained,
        moge2_intrinsics=moge2_intrinsics,
    )
    if intrinsics_source == "moge2":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using MoGe-2 estimated intrinsics in original space: fx={frame_fx_org:.2f}, fy={frame_fy_org:.2f}, cx={frame_cx_org:.2f}, cy={frame_cy_org:.2f}"
        )
    elif intrinsics_source == "default":
        print(
            "Camera intrinsics are partially/fully missing. "
            f"Using image-size defaults in original space: fx={frame_fx_org:.2f}, fy={frame_fy_org:.2f}, cx={frame_cx_org:.2f}, cy={frame_cy_org:.2f}"
        )

    gt = None if gt_depth is None else depth_to_disparity(gt_depth)
    prompt = None if prompt_depth is None else depth_to_disparity(prompt_depth)

    _, _, h, w = image.shape
    fx, fy, cx, cy, _ = build_scaled_intrinsics_matrix(
        fx_org=frame_fx_org,
        fy_org=frame_fy_org,
        cx_org=frame_cx_org,
        cy_org=frame_cy_org,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        device=image.device,
    )
    print(f"Scaled Intrinsics: fx {fx:.2f}, fy {fy:.2f}, cx {cx:.2f}, cy {cy:.2f}")

    sky_mask = run_optional_sky_mask(
        image=image,
        enable_skyseg_model=args.enable_skyseg_model,
        sky_model_ckpt_path=args.sky_model_ckpt_path,
    )

    h_sample, w_sample = resolve_output_size_from_mode(
        output_resolution_mode=args.output_resolution_mode,
        org_h=org_h,
        org_w=org_w,
        h=h,
        w=w,
        output_size=args.output_size,
        upsample_ratio=args.upsample_ratio,
    )

    query_2d_uniform_coord = SAMPLING_METHODS["2d_uniform"]((h_sample, w_sample)).unsqueeze(0).to(device)
    pred_2d_uniform_depth, _ = model.inference(
        image=image,
        query_coord=query_2d_uniform_coord,
        gt_depth=gt,
        gt_depth_mask=gt_depth_mask,
        prompt_depth=prompt,
        prompt_mask=None if prompt is None else prompt > 0,
    )
    pred_depthmap = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, h_sample, w_sample)

    pred_depthmap, pred_2d_uniform_depth = apply_sky_mask_to_depth(
        pred_depthmap=pred_depthmap,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        sky_mask=sky_mask,
        h_sample=h_sample,
        w_sample=w_sample,
        sky_depth_value=200.0,
    )

    return DepthInferenceResult(
        input_image_path=frame_image_path,
        org_img=org_img,
        image=image,
        query_2d_uniform_coord=query_2d_uniform_coord,
        pred_2d_uniform_depth=pred_2d_uniform_depth,
        pred_depthmap=pred_depthmap,
        org_h=org_h,
        org_w=org_w,
        input_h=h,
        input_w=w,
        output_h=h_sample,
        output_w=w_sample,
        fx_org=frame_fx_org,
        fy_org=frame_fy_org,
        cx_org=frame_cx_org,
        cy_org=frame_cy_org,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        intrinsics_source=intrinsics_source,
    )


def _to_single_depth_tensor(data, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(data, device=device, dtype=dtype)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor.unsqueeze(1)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(f"Expected depth-like tensor with shape [H,W], [1,H,W], or [1,1,H,W], got {tuple(tensor.shape)}")


def scale_align_depth_result(
    result: DepthInferenceResult,
    reference_depth,
    *,
    reference_conf=None,
    confidence_threshold: float = 0.0,
    min_valid_pixels: int = 128,
    trim_quantile: float = 0.05,
) -> tuple[float, int]:
    pred_depth = result.pred_depthmap
    ref_depth = _to_single_depth_tensor(reference_depth, device=pred_depth.device, dtype=pred_depth.dtype)
    ref_depth = F.interpolate(
        ref_depth,
        size=(result.output_h, result.output_w),
        mode="bilinear",
        align_corners=False,
    )

    valid = torch.isfinite(pred_depth) & torch.isfinite(ref_depth) & (pred_depth > 1e-6) & (ref_depth > 1e-6)

    if reference_conf is not None:
        ref_conf = _to_single_depth_tensor(reference_conf, device=pred_depth.device, dtype=pred_depth.dtype)
        ref_conf = F.interpolate(
            ref_conf,
            size=(result.output_h, result.output_w),
            mode="bilinear",
            align_corners=False,
        )
        valid &= torch.isfinite(ref_conf) & (ref_conf > float(confidence_threshold))

    valid_count = int(valid.sum().item())
    result.depth_scale_align_valid_pixels = valid_count
    if valid_count < int(min_valid_pixels):
        print(f"[Warning] Skip DA3 scale alignment: only {valid_count} valid pixels.")
        return 1.0, valid_count

    ratios = (ref_depth[valid] / pred_depth[valid]).flatten()
    ratios = ratios[torch.isfinite(ratios) & (ratios > 0)]
    if ratios.numel() < int(min_valid_pixels):
        valid_count = int(ratios.numel())
        result.depth_scale_align_valid_pixels = valid_count
        print(f"[Warning] Skip DA3 scale alignment: only {valid_count} positive ratios.")
        return 1.0, valid_count

    if 0.0 < trim_quantile < 0.5 and ratios.numel() > 2:
        lower = torch.quantile(ratios, trim_quantile)
        upper = torch.quantile(ratios, 1.0 - trim_quantile)
        trimmed = ratios[(ratios >= lower) & (ratios <= upper)]
        if trimmed.numel() > 0:
            ratios = trimmed

    scale = float(torch.median(ratios).item())
    if not np.isfinite(scale) or scale <= 0:
        print(f"[Warning] Skip DA3 scale alignment: invalid scale {scale}.")
        return 1.0, valid_count

    result.pred_depthmap = result.pred_depthmap * scale
    result.pred_2d_uniform_depth = result.pred_2d_uniform_depth * scale
    result.depth_scale_align_factor *= scale
    return scale, valid_count


def build_point_cloud_from_depth_result(
    result: DepthInferenceResult,
    *,
    pcd_extrinsics_w2c: Optional[np.ndarray] = None,
    pcd_intrinsics_override: Optional[np.ndarray] = None,
    filter_flying_points: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
):
    pcd_intrinsics = result.output_intrinsics_matrix()
    if pcd_intrinsics_override is not None:
        pcd_intrinsics = np.asarray(pcd_intrinsics_override, dtype=np.float32)
        if pcd_intrinsics.shape != (3, 3):
            raise ValueError(
                f"pcd_intrinsics_override must have shape (3, 3), got {pcd_intrinsics.shape}"
            )
    pcd = depth2pcd(
        result.query_2d_uniform_coord.squeeze().cpu(),
        result.pred_2d_uniform_depth.squeeze().cpu(),
        result.image.squeeze().cpu(),
        pcd_intrinsics,
        ext=pcd_extrinsics_w2c,
    )
    if filter_flying_points and len(pcd.points) > 0:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        pcd = pcd.select_by_index(ind)
    return pcd


def save_depth_inference_result(
    result: DepthInferenceResult,
    *,
    depth_vis_path: str,
    depth_raw_path: Optional[str] = None,
    pcd_path: Optional[str] = None,
    save_pcd: bool = True,
    pcd_extrinsics_w2c: Optional[np.ndarray] = None,
    pcd_intrinsics_override: Optional[np.ndarray] = None,
):
    plot_depth(result.org_img, result.pred_depthmap, depth_vis_path)
    if depth_raw_path is not None:
        save_depth_array(result.pred_depthmap, depth_raw_path)

    if not save_pcd:
        return None

    if pcd_path is not None:
        return save_sampled_point_clouds(
            result.query_2d_uniform_coord.squeeze().cpu(),
            result.pred_2d_uniform_depth.squeeze().cpu(),
            result.image.squeeze().cpu(),
            result.fx,
            result.fy,
            result.cx,
            result.cy,
            pcd_path,
            ixt=pcd_intrinsics_override,
            extrinsics_w2c=pcd_extrinsics_w2c,
        )

    return build_point_cloud_from_depth_result(
        result,
        pcd_extrinsics_w2c=pcd_extrinsics_w2c,
        pcd_intrinsics_override=pcd_intrinsics_override,
    )


@torch.no_grad()
def main(args: DepthInferenceArgs) -> None:
    model, device = load_depth_model(args)
    result = run_depth_inference(args, model=model, device=device)

    output_paths = resolve_depth_output_paths(
        input_image_path=args.input_image_path,
        model_type=args.model_type,
        output_resolution_mode=args.output_resolution_mode,
        upsample_ratio=args.upsample_ratio,
        h_sample=result.output_h,
        w_sample=result.output_w,
        depth_output_dir=args.depth_output_dir,
        pcd_output_dir=args.pcd_output_dir,
    )

    save_depth_inference_result(
        result,
        depth_vis_path=output_paths.depth_path,
        pcd_path=output_paths.pcd_path if args.save_pcd else None,
        save_pcd=args.save_pcd,
    )


if __name__ == "__main__":
    main(tyro.cli(DepthInferenceArgs))
