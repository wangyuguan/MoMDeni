"""The distribution of viewing directions.

The in-plane angle is uniform, so the distribution is a density on the sphere of viewing directions. :func:`viewing_grid` gives a quadrature
grid on S^2 with positive weights, and :class:`LowDegreeViewingDensity` a positive density on it: a sum of squares of real spherical
harmonics of degree 0, 2 or 4, symmetric under v -> -v.
"""

from __future__ import annotations

import os
import re

import numpy as np
from scipy.special import gammaln, lpmv


class ViewingGrid:
    """Viewing directions of an S^2 quadrature (the in-plane angle is integrated out by the FB moments).

    Attributes
    ----------
    rots : (Ns, 3, 3)    one rotation per node, with the node direction as its third column
    Ns : int             number of nodes
    q : (Ns,)            probability masses for uniform surface measure
    xyz : (Ns, 3)        node directions (unit vectors)
    """

    def __init__(self, rots, q, xyz):
        self.rots = rots
        self.Ns = len(q)
        self.q = q
        self.xyz = xyz


SPHERE_RULE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "sphere_rules")
_RULE_CACHE = {}


def _sphere_rules():
    """Load and cache the precomputed S^2 quadrature catalogue.

    Files contain rows ``x y z w`` after two header lines. Nodes are normalized
    to the unit sphere and weights to unit sum. Each rule records whether all
    weights are positive. The returned catalogue is sorted by degree.
    """
    if _RULE_CACHE:
        return _RULE_CACHE["rules"]
    rules = []
    for path in sorted(os.listdir(SPHERE_RULE_DIR)):
        m = re.match(r"N(\d+)_M(\d+)_(\w+)\.dat$", path)
        if not m:
            continue
        d = np.genfromtxt(os.path.join(SPHERE_RULE_DIR, path), skip_header=2)
        xyz, w = np.atleast_2d(d)[:, :3], np.atleast_1d(np.atleast_2d(d)[:, 3])
        rules.append({"degree": int(m.group(1)), "n": w.size,
                      "symmetry": m.group(3), "file": path,
                      "xyz": xyz / np.linalg.norm(xyz, axis=1, keepdims=True),
                      "q": w / w.sum(), "positive": bool(np.all(w > 0))})
    rules.sort(key=lambda r: r["degree"])
    _RULE_CACHE["rules"] = rules
    return rules


def sphere_quadrature(degree):
    """Select the smallest positive-weight Gauss quadrature on S^2 of at least this polynomial degree.

    Rules with negative weights are excluded because the weights define a
    probability mass over viewing directions.

    Returns polar angles, azimuths, normalized weights, and unit directions.
    """
    usable = [r for r in _sphere_rules() if r["positive"] and r["degree"] >= degree]
    rule = min(usable, key=lambda r: r["n"])
    xyz = rule["xyz"]
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, phi, rule["q"].copy(), xyz.copy()


def viewing_grid(degree):
    """The viewing directions of :func:`sphere_quadrature` as rotations with the node direction as third column."""
    th, ph, q, xyz = sphere_quadrature(degree)
    rots = np.empty((th.size, 3, 3))
    for a in range(th.size):
        rots[a] = Rz(ph[a]) @ Ry(th[a])
    return ViewingGrid(rots, q, xyz)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])




def real_spherical_harmonics(directions, degree):
    """Real orthonormal SH columns ordered by ``ell=1..degree`` and ``m``.

    The constant harmonic is omitted.  Orthogonality uses surface measure
    with total area ``4*pi``.
    """

    directions = np.asarray(directions, dtype=float)
    phi = np.arctan2(directions[:, 1], directions[:, 0])
    cosine = np.clip(directions[:, 2], -1.0, 1.0)
    columns = []
    for ell in range(1, int(degree) + 1):
        for m in range(-ell, ell + 1):
            order = abs(m)
            norm = np.sqrt(
                (2 * ell + 1) / (4 * np.pi)
                * np.exp(gammaln(ell - order + 1)
                         - gammaln(ell + order + 1)))
            radial = norm * lpmv(order, ell, cosine)
            if m == 0:
                value = radial
            else:
                trig = np.sin(order * phi) if m < 0 else np.cos(order * phi)
                value = np.sqrt(2.0) * (-1.0) ** order * radial * trig
            columns.append(value)
    if columns:
        return np.column_stack(columns)
    return np.empty((len(directions), 0))


class LowDegreeViewingDensity:
    """Positive full-real-SH sum-of-squares density relative to Haar.

    Features contain the constant and all real spherical harmonics through
    half of ``degree``.  The coefficient matrix is identity plus free entries
    below its first row, which fixes the irrelevant overall scale.

    Only coefficients that couple features of equal parity are kept.  Each
    squared row then has a definite parity, so the density carries even
    spherical-harmonic degrees alone and satisfies ``rho(-v) = rho(v)``.
    This is the exact symmetry of reflection-augmented moments: a mirrored
    image is the projection along ``-v``, so the odd degrees are
    unidentifiable flat directions of that objective.  Degree 0, 2, and 4
    then have 0, 6, and 26 free parameters.
    """

    def __init__(self, grid, degree=2):
        self.grid = grid
        self.degree = int(degree)
        self.H = np.column_stack((
            np.ones(len(grid.xyz)),
            np.sqrt(4 * np.pi)
            * real_spherical_harmonics(grid.xyz, self.degree // 2),
        ))
        self.n_features = self.H.shape[1]
        q = np.asarray(grid.q, dtype=np.float64)
        self.q = q.copy()
        self.gram = self.H.T @ (self.q[:, None] * self.H)
        self.gram_error = float(
            np.max(np.abs(self.gram - np.eye(self.n_features))))
        self.feature_parity = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.concatenate([np.full(2 * ell + 1, ell % 2, dtype=np.int64)
                            for ell in range(1, self.degree // 2 + 1)])
            if self.degree else np.zeros(0, dtype=np.int64),
        ])
        rows, cols = np.tril_indices(self.n_features)
        rows, cols = rows[1:], cols[1:]
        same = self.feature_parity[rows] == self.feature_parity[cols]
        self.rows, self.cols = rows[same], cols[same]
        self.n_parameters = len(self.rows)
        self._moment_features = np.column_stack((
            np.ones(grid.Ns),
            np.sqrt(4 * np.pi)
            * real_spherical_harmonics(grid.xyz, self.degree),
        ))

    def _components(self, beta):
        matrix = np.eye(self.n_features)
        matrix[self.rows, self.cols] += np.asarray(beta, dtype=np.float64)
        values = self.H @ matrix.T
        power = np.sum(values**2, axis=1)
        trace = float(np.sum(matrix**2))
        normalization = float(self.q @ power)
        return values, power, trace, normalization

    def density(self, beta, directions):
        """Evaluate the SOS density relative to uniform sphere measure at the unit vectors ``directions``.

        Divide by ``4*pi`` for density per steradian.
        """

        _, _, trace, _ = self._components(beta)
        harmonics = real_spherical_harmonics(directions, self.degree // 2)
        features = np.column_stack((
            np.ones(len(harmonics)), np.sqrt(4 * np.pi) * harmonics,
        ))
        matrix = np.eye(self.n_features)
        matrix[self.rows, self.cols] += np.asarray(beta, dtype=np.float64)
        power = np.sum((features @ matrix.T)**2, axis=1)
        return power / trace

    def pmf(self, beta):
        """Return normalized masses including the sphere quadrature weights."""

        _, power, _, normalization = self._components(beta)
        return self.q * power / normalization

    def vjp(self, beta, grad_p):
        """Apply the transpose derivative of :meth:`pmf`."""

        grad_p = np.asarray(grad_p, dtype=np.float64)
        values, power, _, normalization = self._components(beta)
        p = self.q * power / normalization
        weights = self.q * (grad_p - float(p @ grad_p)) / normalization
        matrix_gradient = 2 * values.T @ (weights[:, None] * self.H)
        return matrix_gradient[self.rows, self.cols]

    def evaluate(self, beta):
        """Return normalization, concentration, and SH diagnostics."""

        beta = np.asarray(beta, dtype=np.float64)
        _, power, trace, normalization = self._components(beta)
        relative_density = power / trace
        p = self.q * power / normalization
        moments = p @ self._moment_features
        moment_power = [
            float(np.sum(moments[ell * ell:(ell + 1) ** 2] ** 2))
            for ell in range(self.degree + 1)
        ]
        return dict(
            viewing_degree=self.degree,
            n_view_parameters=self.n_parameters,
            viewing_feature_degree=self.degree // 2,
            viewing_n_features=self.n_features,
            viewing_coefficient_norm=float(np.linalg.norm(beta)),
            viewing_density_min=float(relative_density.min()),
            viewing_density_max=float(relative_density.max()),
            viewing_density_upper_bound=self.n_features,
            viewing_density_lower_bound=1.0 / trace,
            viewing_matrix_norm_squared=trace,
            viewing_quadrature_normalization=normalization / trace,
            viewing_quadrature_gram_error=self.gram_error,
            viewing_mass_sum=float(p.sum()),
            viewing_min_mass=float(p.min()),
            viewing_max_mass=float(p.max()),
            viewing_ess=float(1.0 / np.sum(p * p)),
            viewing_sh_moments=moments.tolist(),
            viewing_sh_power_by_degree=moment_power,
        )
