"""Point-group symmetry of the volume as a linear constraint on its SphFB coefficients.

A rotation ``g`` acts on the coefficients ``a`` of every ``(l, k)`` block through the Wigner matrix ``D^l(g)`` and does not touch ``k``, so a
volume with symmetry group ``G`` satisfies ``D^l(g) a = a`` for the generators of ``G``. In the real coefficients ``theta`` (``a = S theta``) this reads
``(O^l(g) - I) theta_{lk} = 0`` with the real matrix ``O^l(g) = S^-1 D^l(g) S``. The solution space is the same for every ``k``; ``invariant_basis``
returns an orthonormal basis ``B`` of it, so that every symmetric volume is ``theta = B @ phi``.

The axes are fixed (the volume frame of the moments is arbitrary): ``Cn`` is a rotation by 2 pi / n about z; ``Dn`` adds a half turn about x;
``T`` and ``O`` have a 3-fold axis along (1, 1, 1) and 2-fold (T) or 4-fold (O) axes along z; ``I`` adds a 5-fold axis along (0, 1, golden ratio) to the
generators of ``T`` (2-fold axes along x, y, z).
"""

import re

import numpy as np

_GOLDEN = (1 + np.sqrt(5)) / 2


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def generators(name):
    """Rotation matrices that generate the point group ``name`` (``Cn``, ``Dn``, ``T``, ``O`` or ``I``)."""
    z, x = (0, 0, 1), (1, 0, 0)
    cyclic, dihedral = re.fullmatch(r"C(\d+)", name), re.fullmatch(r"D(\d+)", name)
    if cyclic:
        return [_rotation(z, 2 * np.pi / int(cyclic.group(1)))]
    if dihedral:
        return [_rotation(z, 2 * np.pi / int(dihedral.group(1))), _rotation(x, np.pi)]
    three = _rotation((1, 1, 1), 2 * np.pi / 3)
    return {"T": [_rotation(z, np.pi), three],
            "O": [_rotation(z, np.pi / 2), three],
            "I": [_rotation(z, np.pi), three, _rotation((0, 1, _GOLDEN), 2 * np.pi / 5)]}[name]


def invariant_basis(param, name):
    """Orthonormal basis ``B`` (``n_params x n_free``) of the volume coefficient vectors that are symmetric under ``name``; see the module doc."""
    from .optimization import _rot_to_zyz, _wigner_d

    angles = [_rot_to_zyz(g) for g in generators(name)]
    null = []
    for ell in range(param.ell_max + 1):
        rows = [param.indices[(ell, 0, m)] for m in range(-ell, ell + 1)]
        S = param.S[np.ix_(rows, rows)]
        conditions = [np.linalg.solve(S, _wigner_d(ell, *a) @ S).real - np.eye(2 * ell + 1) for a in angles]
        _, s, vh = np.linalg.svd(np.vstack(conditions))
        null.append(vh[int((s > 1e-8).sum()):].T)
    basis = np.zeros((param.n_params, sum(param.k_max[ell] * null[ell].shape[1] for ell in range(param.ell_max + 1))))
    column = 0
    for ell in range(param.ell_max + 1):
        for k in range(param.k_max[ell]):
            rows = [param.indices[(ell, k, m)] for m in range(-ell, ell + 1)]
            basis[rows, column:column + null[ell].shape[1]] = null[ell]
            column += null[ell].shape[1]
    return basis
