"""Backward-compatible subset training CLI wrapper."""

from .training.subset import main, run_subset_training

__all__ = ["main", "run_subset_training"]


if __name__ == "__main__":
    main()

