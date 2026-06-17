"""Backward-compatible smoke test module."""

from importlib import import_module


def main() -> None:
    smoke = import_module("src.tests.test_smoke")
    evaluation = import_module("src.tests.test_evaluation")
    smoke.test_projection_crop_shift()
    smoke.test_identity_voxel_metrics()
    smoke.test_smoke_shapes()
    evaluation.test_surface_identity()
    print("smoke tests passed")


if __name__ == "__main__":
    main()

