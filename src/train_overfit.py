"""Backward-compatible overfit CLI wrapper."""

from .training.overfit import main, run_overfit

__all__ = ["main", "run_overfit"]


if __name__ == "__main__":
    main()

