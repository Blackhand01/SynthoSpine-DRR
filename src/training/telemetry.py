"""Runtime telemetry helpers."""

from __future__ import annotations

import gc
import time


def now() -> float:
    return time.perf_counter()


def collect_runtime(start_time: float, items_processed: int = 1) -> dict[str, float]:
    elapsed = max(time.perf_counter() - start_time, 1e-8)
    return {
        "seconds": elapsed,
        "throughput_items_sec": items_processed / elapsed,
        "ram_mb": _ram_mb(),
        "vram_mb": _vram_mb(),
    }


def cleanup() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _ram_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss / (1024 * 1024))
    except ImportError:
        return 0.0


def _vram_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.memory_allocated() / (1024 * 1024))
    except ImportError:
        pass
    return 0.0

