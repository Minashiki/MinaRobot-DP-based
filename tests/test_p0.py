"""Numerical and reproducibility regressions for the frozen P0 contract."""

from feedingrobot.scripts.validate_p0 import check_ddim, check_hdf5, check_rng


def test_ddim_sample_direction_and_internal_clipping():
    check_ddim()


def test_independent_rng_and_resume():
    check_rng()


def test_hdf5_uint8_and_float32_roundtrip():
    check_hdf5()
