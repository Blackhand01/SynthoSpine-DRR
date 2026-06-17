"""Backward-compatible smoke CLI wrapper."""

from .apps.smoke import main, run_smoke

__all__ = ["main", "run_smoke"]


if __name__ == "__main__":
    main()

