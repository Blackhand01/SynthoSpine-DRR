"""Backward-compatible evaluation CLI wrapper."""

from .evaluation.runner import main, run_evaluation

__all__ = ["main", "run_evaluation"]


if __name__ == "__main__":
    main()

