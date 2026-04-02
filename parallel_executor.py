import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .progress_logger import ProgressLogger
else:
    try:
        from .progress_logger import ProgressLogger
    except ImportError:  # Allow running as a script without package context
        from progress_logger import ProgressLogger

TVAR = TypeVar("TVAR")
RVAR = TypeVar("RVAR")

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    Generic parallel executor for running functions with items in parallel.
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker threads. If None, uses CPU count.
        """
        self.max_workers = max_workers

    def run_in_parallel(
        self,
        function: Callable[..., RVAR],
        item_list: Sequence[TVAR],
        progress_desc: str = "Processing",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> list[RVAR | None]:
        """
        Execute a function in parallel for each item.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.

        Returns:
            List of results from the function calls (in same order as item_list).
        """
        if not item_list:
            return []

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        results: list[RVAR | None] = [None] * len(item_list)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks and remember index for each future (order must match item_list)
            future_to_idx: dict = {}
            for idx, item in enumerate(item_list):
                future = executor.submit(function, item, **kwargs)
                future_to_idx[future] = idx

            # Process completed tasks with progress logger; store by index so order matches item_list
            every_n = max(1, len(item_list) // 10)
            with ProgressLogger(len(item_list), every_n=every_n, desc=progress_desc, logger=logger) as progress:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as exc:
                        logger.warning(
                            "Processing item %s generated an exception: %s",
                            item_list[idx],
                            exc,
                        )
                        results[idx] = None
                    progress.update(1)

        return results

    def run_in_parallel_no_return(
        self,
        function: Callable[..., Any],
        item_list: Sequence[TVAR],
        progress_desc: str = "Processing",
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Execute a function in parallel for each item without collecting results.
        More memory efficient when you don't need the return values.

        Args:
            function: Function to execute. Should accept (item, **kwargs) as arguments.
            item_list: List of items to process.
            progress_desc: Description for the progress bar.
            max_workers: Override the default max_workers for this execution.
            **kwargs: Additional keyword arguments to pass to the function.
        """
        if not item_list:
            return

        # Determine number of workers
        workers = max_workers or self.max_workers
        if workers is None:
            workers = min(len(item_list), os.cpu_count() or 1)

        print(f"    {progress_desc}: {len(item_list)} items using {workers} workers...")

        import time

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_item = {executor.submit(function, item, **kwargs): item for item in item_list}

            # Process completed tasks with progress logger
            every_n = max(1, len(item_list) // 10)
            with ProgressLogger(len(item_list), every_n=every_n, desc=progress_desc, logger=logger) as progress:
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.warning("Processing item %s generated an exception: %s", item, exc)
                    progress.update(1)

        elapsed = time.time() - start_time
        print(f"    Completed {progress_desc} in {elapsed:.2f} seconds ({elapsed / len(item_list):.2f}s per item)")
