"""Golden-vector tests for the native mix_seed hash and its stream ids."""

import pytest

try:
    from shrimp import _rust as shrimp_rust
except ImportError:  # pragma: no cover
    shrimp_rust = None

needs_native = pytest.mark.skipif(shrimp_rust is None, reason="native module not built")


@needs_native
def test_mix_seed_golden_vectors() -> None:
    assert shrimp_rust.mix_seed(0, 0, 0, 0) == 0x7DE5_3DE7_72EA_694C
    assert shrimp_rust.mix_seed(1, 2, 3, 4) == 0xA6A9_B091_CFF9_D67A
    a = shrimp_rust.mix_seed(123_456_789, 987_654_321, 42, 0)
    b = shrimp_rust.mix_seed(123_456_789, 987_654_321, 42, 1)
    assert a != b  # different stream index yields a different value
