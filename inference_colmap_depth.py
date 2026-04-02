"""
Run InfiniDepth on a COLMAP frame: sparse SfM depth from projected 3D points, then dense inference.

Expects:
  {scene_folder}/images
  {scene_folder}/sparse

Usage:
  python inference_colmap_depth.py -sf /path/to/scene -fidx 0 -o pred_depth \\
      [--model-type InfiniDepth_DepthSensor ...]

``-fidx`` selects the n-th image when images are sorted by file name (basename).
Outputs are written under ``{scene_folder}/{output_rel}/``.

Optional ``-df`` / ``--dmap-folder``: load ``{scene_folder}/{dmap_folder}/...`` ``.npy`` or ``.tif`` / ``.tiff``
depth per image instead of COLMAP sparse depth when a matching file exists (see ``try_load_depth_map_file``).

For multiple frames in code, load ``ColmapInterface`` and the depth model once, then call
``run_colmap_frame_depth_inference`` in a loop (reuse ``dmap_folder=...`` when using file depth).
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from colmap_interface import ColmapInterface
from inference_depth import DepthInferenceArgs, load_depth_model, run_depth_inference
from InfiniDepth.utils.io_utils import plot_depth, save_depth_array, save_sampled_point_clouds

import logging
logger = logging.getLogger(__name__)


def _sorted_image_ids(colmap: ColmapInterface) -> list[int]:
    return sorted(colmap.recon.images.keys(), key=lambda iid: colmap.recon.images[iid].name)


def build_sparse_metric_depth_map(
    colmap: ColmapInterface,
    image_id: int,
    image_h: int,
    image_w: int,
) -> np.ndarray:
    """
    Project COLMAP points3D visible in ``image_id`` into the image plane and rasterize
    metric Z depth (camera-frame). Pixels with no projection stay 0 (invalid for depth loaders).
    When multiple points map to the same pixel, the smallest depth (closest) is kept.
    """
    colmap.ensure_image_point_ids_cached()
    info = colmap.get_image_info(image_id)
    point_ids: list[int] = info["point_ids"]
    cam_h = int(info["height"])
    cam_w = int(info["width"])
    if cam_h <= 0 or cam_w <= 0:
        raise ValueError(f"Invalid COLMAP camera size {(cam_w, cam_h)} for image_id={image_id}")

    R = np.asarray(info["R"], dtype=np.float64)
    t = np.asarray(info["t"], dtype=np.float64).reshape(3)
    K = np.asarray(info["K"], dtype=np.float64)
    dist = info["distortion_params"]
    dist_np = None if dist is None or len(dist) == 0 else np.asarray(dist, dtype=np.float64).reshape(-1)

    sparse = np.zeros((image_h, image_w), dtype=np.float32)
    if not point_ids:
        logger.warning(f"No points found for image_id={image_id} during sparse depth map construction.")
        return sparse

    xyz = np.array([colmap.recon.points3D[pid].xyz for pid in point_ids], dtype=np.float64)
    Pc = np.einsum("ij,nj->ni", R, xyz) + t
    Z = Pc[:, 2]

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)
    imgpts, _ = cv2.projectPoints(xyz, rvec, tvec, K, dist_np)
    imgpts = imgpts.reshape(-1, 2)

    su = image_w / float(cam_w)
    sv = image_h / float(cam_h)
    u = np.round(imgpts[:, 0] * su).astype(np.int32)
    v = np.round(imgpts[:, 1] * sv).astype(np.int32)

    valid = (
        (Z > 1e-6)
        & (u >= 0)
        & (u < image_w)
        & (v >= 0)
        & (v < image_h)
    )
    idxs = np.nonzero(valid)[0]
    for i in idxs:
        vi, ui = int(v[i]), int(u[i])
        z = float(Z[i])
        prev = float(sparse[vi, ui])
        if prev <= 0.0 or z < prev:
            sparse[vi, ui] = z

    return sparse


def _world_to_cam_4x4(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return T


def _read_depth_map_npy_or_tiff(path: Path) -> np.ndarray:
    """Load a single HxW depth array from ``.npy`` (numeric ndarray only) or ``.tif`` / ``.tiff``."""
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path, allow_pickle=False)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Expected ndarray in {path}, got {type(arr).__name__}")
        if arr.dtype == object:
            raise TypeError(f"Unsupported object-dtype array in {path}")
        d = np.asarray(arr, dtype=np.float32)
    elif suf in (".tif", ".tiff"):
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise ValueError(f"Failed to read TIFF: {path}")
        if im.ndim == 3:
            d = im[:, :, 0].astype(np.float32)
        else:
            d = im.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth file extension (use .npy, .tif, .tiff): {path}")

    d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"Expected HxW depth in {path}, got shape {d.shape}")
    return d


def _iter_depth_map_candidate_paths(dmap_root: Path, image_name: str):
    """Search order: mirrored path with each ext, then ``stem.ext``, then ``name.ext`` (per ext)."""
    rel = Path(image_name)
    exts = (".npy", ".tif", ".tiff")
    for ext in exts:
        yield dmap_root / rel.with_suffix(ext)
    for ext in exts:
        yield dmap_root / f"{rel.stem}{ext}"
    for ext in exts:
        yield dmap_root / f"{rel.name}{ext}"


def try_load_depth_map_file(
    scene_folder: Path,
    dmap_folder: str | Path,
    image_name: str,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray | None, Path | None]:
    """
    Look for a depth file under ``scene_folder / dmap_folder``.

    Supported formats: ``.npy`` (plain numeric ndarray, HxW after squeeze) or ``.tif`` / ``.tiff``.

    Tries paths from ``_iter_depth_map_candidate_paths``; first existing file wins.
    Returns ``(depth_hw_float32, path_used)`` or ``(None, None)`` if none exist.
    """
    root = Path(scene_folder).resolve()
    dmap_root = root / Path(dmap_folder)
    path: Path | None = None
    for c in _iter_depth_map_candidate_paths(dmap_root, image_name):
        if c.is_file():
            path = c
            break
    if path is None:
        return None, None

    d = _read_depth_map_npy_or_tiff(path)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    if d.shape[0] != target_h or d.shape[1] != target_w:
        d = cv2.resize(d, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return d.astype(np.float32, copy=False), path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COLMAP scene + InfiniDepth single-frame inference.")
    p.add_argument("-sf", "--scene-folder", required=True, help="Scene root (contains images/ and sparse/).")
    p.add_argument(
        "-fidx",
        "--frame-index",
        type=int,
        required=True,
        help="Index into images sorted by COLMAP image file name (0-based).",
    )
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory relative to scene_folder.",
    )
    p.add_argument("--colmap-subdir", type=str, default="sparse", help="COLMAP model folder under scene_folder.")
    p.add_argument("--images-subdir", type=str, default="images", help="Images folder under scene_folder.")
    p.add_argument("--n-min-tracks", type=int, default=3, help="Min track length for active COLMAP points.")

    p.add_argument("--model-type", type=str, default="InfiniDepth_DepthSensor")
    p.add_argument("--depth-model-path", type=str, default="checkpoints/depth/infinidepth_depthsensor.ckpt")
    p.add_argument("--moge2-pretrained", type=str, default="checkpoints/moge-2-vitl-normal/model.pt")
    p.add_argument("--input-size", type=int, nargs=2, default=[768, 1024], metavar=("H", "W"))
    p.add_argument("--output-size", type=int, nargs=2, default=[768, 1024], metavar=("H", "W"))
    p.add_argument(
        "--output-resolution-mode",
        type=str,
        default="upsample",
        choices=["upsample", "original", "specific"],
    )
    p.add_argument("--upsample-ratio", type=int, default=1)
    p.add_argument("--enable-skyseg-model", action="store_true")
    p.add_argument("--sky-model-ckpt-path", type=str, default="checkpoints/sky/skyseg.onnx")
    p.add_argument("--no-filter-pcd", action="store_true", help="Disable statistical outlier removal on the PLY.")
    p.add_argument(
        "-df",
        "--dmap-folder",
        type=str,
        default=None,
        help=(
            "Optional folder under scene_folder with per-image depth .npy files "
            "(see try_load_depth_map_file). If set, loads that depth instead of COLMAP sparse depth when found."
        ),
    )
    return p.parse_args()


def depth_inference_args_from_argparse(ns: argparse.Namespace) -> DepthInferenceArgs:
    """Build ``DepthInferenceArgs`` shared across frames (intrinsics set per frame in ``run_colmap_frame_depth_inference``)."""
    return DepthInferenceArgs(
        input_image_path="",
        input_depth_path=None,
        model_type=ns.model_type,
        depth_model_path=ns.depth_model_path,
        moge2_pretrained=ns.moge2_pretrained,
        input_size=(int(ns.input_size[0]), int(ns.input_size[1])),
        output_size=(int(ns.output_size[0]), int(ns.output_size[1])),
        output_resolution_mode=ns.output_resolution_mode,
        upsample_ratio=int(ns.upsample_ratio),
        fx_org=None,
        fy_org=None,
        cx_org=None,
        cy_org=None,
        enable_skyseg_model=bool(ns.enable_skyseg_model),
        sky_model_ckpt_path=ns.sky_model_ckpt_path,
        save_pcd=False,
    )


@torch.no_grad()
def run_colmap_frame_depth_inference(
    colmap: ColmapInterface,
    *,
    image_id: int,
    out_dir: Path | str,
    depth_args: DepthInferenceArgs,
    model: torch.nn.Module,
    device: torch.device,
    filter_flying_points: bool = True,
    verbose: bool = True,
    dmap_folder: str | Path | None = None,
) -> dict[str, str]:
    """
    Run depth inference for the ``image_id``.

    ``colmap``, ``model``, and ``device`` are expected to be created once by the caller.
    ``depth_args`` is a template; per-frame intrinsics and ``input_image_path`` are applied internally.

    If ``dmap_folder`` is set, loads ``scene_folder/dmap_folder/...`` ``.npy`` / ``.tif`` / ``.tiff`` for this image
    (see ``try_load_depth_map_file``); when the file exists it replaces COLMAP sparse depth.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = colmap.get_image_info(image_id)
    image_path = Path(info["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file missing for image_id={image_id}: {image_path}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_h, image_w = bgr.shape[0], bgr.shape[1]

    K = np.asarray(info["K"], dtype=np.float64)
    cam_w = float(info["width"])
    cam_h = float(info["height"])
    su = image_w / cam_w
    sv = image_h / cam_h
    fx_org = float(K[0, 0] * su)
    fy_org = float(K[1, 1] * sv)
    cx_org = float(K[0, 2] * su)
    cy_org = float(K[1, 2] * sv)

    input_depth = None
    depth_source = None
    if dmap_folder is not None:
        loaded, loaded_path = try_load_depth_map_file(
            colmap.workfolder,
            dmap_folder,
            info["image_name"],
            image_h,
            image_w,
        )
        if loaded is not None and loaded_path is not None:
            input_depth = loaded
            depth_source = f"file:{loaded_path}"
            if verbose:
                logger.info("Using depth map from %s (instead of COLMAP sparse depth).", loaded_path)
        else:
            logger.warning(
                "dmap_folder=%s set but no .npy/.tif/.tiff found for image_name=%r under %s; using COLMAP sparse depth.",
                dmap_folder,
                info["image_name"],
                colmap.workfolder / Path(dmap_folder),
            )

    if input_depth is None:
        input_depth = build_sparse_metric_depth_map(colmap, image_id, image_h, image_w)
        depth_source = "colmap_sparse"

    if float(np.max(input_depth)) <= 0.0:
        logger.warning(
            "No valid input depth for image_id=%s (%s); source=%s.",
            image_id,
            image_path.name,
            depth_source,
        )

    stem = f"{image_id:04d}_{Path(info['image_name']).stem}"
    depth_npy_path = str(out_dir / f"{stem}_depth.npy")
    depth_png_path = str(out_dir / f"{stem}_depth_vis.png")
    pcd_path = str(out_dir / f"{stem}_pointcloud.ply")

    frame_depth_args = replace(
        depth_args,
        input_image_path=str(image_path),
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
    )

    zmax = float(np.max(input_depth)) if input_depth.size else 0.0
    depth_load_kwargs = {"max_prompt": max(zmax * 2.0, 1.0e3), "min_prompt": 1.0e-4}

    result = run_depth_inference(
        frame_depth_args,
        model=model,
        device=device,
        input_image_path=str(image_path),
        fx_org=fx_org,
        fy_org=fy_org,
        cx_org=cx_org,
        cy_org=cy_org,
        input_depth_map=input_depth,
        depth_load_kwargs=depth_load_kwargs,
    )

    save_depth_array(result.pred_depthmap, depth_npy_path)
    plot_depth(result.org_img, result.pred_depthmap, depth_png_path)

    w2c = _world_to_cam_4x4(info["R"], info["t"])
    save_sampled_point_clouds(
        result.query_2d_uniform_coord.squeeze().cpu(),
        result.pred_2d_uniform_depth.squeeze().cpu(),
        result.image.squeeze().cpu(),
        float(result.fx),
        float(result.fy),
        float(result.cx),
        float(result.cy),
        pcd_path,
        extrinsics_w2c=w2c,
        filter_flying_points=filter_flying_points,
    )

    paths = {"depth_npy": depth_npy_path, "depth_vis": depth_png_path, "pointcloud_ply": pcd_path}
    if verbose:
        print(f"Wrote depth npy: {depth_npy_path}")
        print(f"Wrote depth vis: {depth_png_path}")
        print(f"Wrote point cloud: {pcd_path}")
    return paths


@torch.no_grad()
def main() -> None:
    ns = parse_args()

    scene_path = Path(ns.scene_folder).resolve()
    out_dir = (scene_path / ns.output).resolve()

    colmap = ColmapInterface(
        workfolder=scene_path,
        colmap_folder=Path(ns.colmap_subdir),
        image_folder=Path(ns.images_subdir),
        n_min_tracks=int(ns.n_min_tracks),
    )

    image_ids = _sorted_image_ids(colmap)
    if not image_ids:
        raise RuntimeError("COLMAP reconstruction has no registered images.")
    if int(ns.frame_index) < 0 or int(ns.frame_index) >= len(image_ids):
        raise IndexError(f"frame_index={ns.frame_index} out of range; got {len(image_ids)} images (0..{len(image_ids) - 1}).")

    image_id = image_ids[int(ns.frame_index)]

    depth_args = depth_inference_args_from_argparse(ns)
    model, device = load_depth_model(depth_args)

    run_colmap_frame_depth_inference(
        colmap,
        image_id=image_id,
        out_dir=out_dir,
        depth_args=depth_args,
        model=model,
        device=device,
        filter_flying_points=not bool(ns.no_filter_pcd),
        dmap_folder=ns.dmap_folder,
    )


if __name__ == "__main__":
    main()
