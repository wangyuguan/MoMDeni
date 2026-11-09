"""Low-degree positive viewing distributions for moment reconstruction.

The viewing distribution is represented on oriented :math:`S^2`, with a
uniform in-plane angle.  Squared real spherical-harmonic features give a
positive density of degree 0, 2, or 4 relative to Haar measure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, lpmv

from .moments import fb_objective_and_grad


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

    def _beta(self, beta):
        beta = np.asarray(beta, dtype=np.float64)
        return beta

    def _components(self, beta):
        matrix = np.eye(self.n_features)
        matrix[self.rows, self.cols] += self._beta(beta)
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
        matrix[self.rows, self.cols] += self._beta(beta)
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

        beta = self._beta(beta)
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


def _single_order_weights(lambdas):
    """The loss weights as arrays and the (1-based) order of the single active moment."""
    weights = [np.asarray(value, dtype=float) for value in lambdas]
    order = next(order for order, weight in enumerate(weights, 1) if np.any(weight > 0))
    return weights, order


def fit_single_moment_low_degree(
        param, grid, targets, lambdas, theta0, beta0=None,
        degree=2, maxiter=5000, print_every=100):
    """Fit exactly one moment order over volume and viewing parameters.

    Returns the volume coefficients, viewing parameters and masses, the optimizer result, the relative misfit of
    the active moment and the viewing-density diagnostics.
    """

    targets = tuple(targets)
    weights, order = _single_order_weights(lambdas)
    density = LowDegreeViewingDensity(grid, degree=degree)
    theta0 = np.asarray(theta0, dtype=np.float64)
    ntheta = param.n_params
    if beta0 is None:
        beta0 = np.zeros(density.n_parameters)
    x0 = np.r_[theta0, density._beta(beta0)]
    target_norm = max(
        float(np.linalg.norm(np.ravel(targets[order - 1]))), 1e-30)
    state = dict(nit=0, nfev=0)

    def objective(x):
        p = density.pmf(x[ntheta:])
        value, gtheta, gp, info = fb_objective_and_grad(
            x[:ntheta], p, param, *targets,
            lambdas=weights, return_info=True,
        )
        gradient = np.r_[gtheta, density.vjp(x[ntheta:], gp)]
        state.update(
            f=value,
            gnorm=float(np.linalg.norm(gradient)),
            relative_error=float(info[f"misfit{order}"] / target_norm),
        )
        state["nfev"] += 1
        return value, gradient

    def callback(_):
        state["nit"] += 1
        if print_every and state["nit"] % print_every == 0:
            print(
                f"    [m{order} SOS{density.degree}] it={state['nit']:5d} "
                f"nfev={state['nfev']:5d} f={state['f']:.6e} "
                f"|g|={state['gnorm']:.2e} "
                f"rel m{order}={state['relative_error']:.2e}",
                flush=True,
            )

    options = dict(
        maxiter=int(maxiter), maxfun=max(15000, 20 * int(maxiter)),
        ftol=1e-11, gtol=1e-10, maxcor=100,
    )
    result = minimize(
        objective, x0, jac=True, method="L-BFGS-B", callback=callback,
        options=options,
    )
    x = np.asarray(result.x, dtype=float).copy()
    theta, beta = x[:ntheta].copy(), x[ntheta:].copy()
    p = density.pmf(beta)
    diagnostics = density.evaluate(beta)
    _, _, _, info = fb_objective_and_grad(
        theta, p, param, *targets, lambdas=weights, return_info=True,
    )
    if print_every:
        print(
            f"  [m{order} SOS{density.degree}] f={result.fun:.6e} "
            f"nit={result.nit:5d} nfev={result.nfev:5d} "
            f"{result.message}",
            flush=True,
        )
    return dict(
        theta=theta, beta=beta, p=p, res=result,
        relative_error=float(info[f"misfit{order}"] / target_norm),
        viewing_diagnostics=diagnostics,
    )
