"""Radial basis sizes of the steerable moment basis."""

import pytest

from kam.moments import FBMomentBasis, RADIAL_SINGULAR_THRESHOLD


@pytest.mark.parametrize("side, ell_max, counts", [
    (16, 6, [8, 7, 7, 6, 6, 5, 5]),
    (20, 7, [10, 9, 9, 8, 8, 7, 7, 6]),
    (16, 9, [8, 8, 7, 7, 6, 6, 5, 5, 4, 4]),
])
def test_radial_counts(side, ell_max, counts):
    basis = FBMomentBasis(side, ell_max)
    assert basis.counts.tolist() == counts
    assert basis.n_coefficients == sum(counts)
    assert basis.summary()["radial_singular_threshold"] == RADIAL_SINGULAR_THRESHOLD
