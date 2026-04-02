import logging
import time


class ProgressLogger:
    def __init__(
        self,
        total: int | None,
        every_n: int,
        desc: str = "",
        logger: logging.Logger | None = None,
    ) -> None:
        self.total = total
        self.every_n = max(1, int(every_n))
        self.desc = desc
        self.logger = logger or logging.getLogger(__name__)
        self.count = 0
        self.start_time = time.perf_counter()
        self.last_log_time = self.start_time

    def __enter__(self) -> "ProgressLogger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def update(self, n: int = 1) -> None:
        self.count += n
        now = time.perf_counter()
        elapsed = now - self.start_time
        self.last_log_time = now
        if self.total is not None:
            if self.count == 1 or self.count % self.every_n == 0 or self.count == self.total:
                rate = self.count / elapsed if elapsed > 0 else 0.0
                remaining = (self.total - self.count) / rate if rate > 0 else float("inf")
                self.logger.info(
                    "   %s: %d/%d (elapsed %.1fs, eta %.1fs)",
                    self.desc,
                    self.count,
                    self.total,
                    elapsed,
                    remaining,
                )
        else:
            if self.count == 1 or self.count % self.every_n == 0:
                rate = self.count / elapsed if elapsed > 0 else 0.0
                self.logger.info("   %s: %d (elapsed %.1fs, rate %.2f/s)", self.desc, self.count, elapsed, rate)

    def close(self) -> None:
        if self.total is None:
            return
        if self.count != self.total:
            elapsed = time.perf_counter() - self.start_time
            self.logger.info("   %s: %d/%d (elapsed %.1fs)", self.desc, self.count, self.total, elapsed)
