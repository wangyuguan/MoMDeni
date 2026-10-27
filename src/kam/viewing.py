"""Low-degree positive viewing distributions for moment reconstruction.

The viewing distribution is represented on oriented :math:`S^2`, with a
uniform in-plane angle.  Squared real spherical-harmonic features give a
positive density of degree 0, 2, or 4 relative to Haar measure.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, lpmv

from .optimization import joint_objective_and_grad


def real_spherical_harmonics(directions, degree):
    """Real orthonormal SH columns ordered by ``ell=1..degree`` and ``m``.

    The constant harmonic is omitted.  Orthogonality uses surface measure
    with total area ``4*pi``.
    """

    if (isinstance(degree, (bool, np.bool_)) or int(degree) != degree
            or degree < 0):
        raise ValueError("degree must be a nonnegative integer")
    directions = np.asarray(directions, dtype=float)
    if (directions.ndim != 2 or directions.shape[1] != 3
            or len(directions) == 0 or not np.all(np.isfinite(directions))
            or not np.allclose(np.linalg.norm(directions, axis=1), 1.0,
                               rtol=0, atol=1e-10)):
        raise ValueError("directions must be finite unit vectors of shape (n, 3)")
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
    below its first row, which fixes the irrelevant overall scale.  Degree 0,
    2, and 4 have 0, 9, and 44 free parameters, respectively.

    ``antipodal`` keeps only coefficients that couple features of equal
    parity.  Each squared row then has a definite parity, so the density
    carries even spherical-harmonic degrees alone and satisfies
    ``rho(-v) = rho(v)``.  This is the exact symmetry of reflection-augmented
    moments: a mirrored image is the projection along ``-v``, so the odd
    degrees are unidentifiable flat directions of that objective.  Degree 0,
    2, and 4 then have 0, 6, and 26 free parameters.
    """

    def __init__(self, grid, degree=2, antipodal=False):
        if isinstance(degree, (bool, np.bool_)) or degree not in (0, 2, 4):
            raise ValueError("degree must be 0, 2, or 4")
        if not isinstance(antipodal, (bool, np.bool_)):
            raise ValueError("antipodal must be Boolean")
        self.grid = grid
        self.degree = int(degree)
        self.antipodal = bool(antipodal)
        self.H = np.column_stack((
            np.ones(len(grid.xyz)),
            np.sqrt(4 * np.pi)
            * real_spherical_harmonics(grid.xyz, self.degree // 2),
        ))
        self.n_features = self.H.shape[1]
        q = np.asarray(grid.q, dtype=np.float64)
        if (q.shape != (grid.Ns,) or len(self.H) != grid.Ns
                or not np.all(np.isfinite(q)) or np.any(q <= 0)
                or not np.isclose(q.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError(
                "grid.q must be finite positive normalized sphere masses")
        self.q = q.copy()
        self.gram = self.H.T @ (self.q[:, None] * self.H)
        self.gram_error = float(
            np.max(np.abs(self.gram - np.eye(self.n_features))))
        if self.gram_error > 1e-10:
            raise ValueError(
                "sphere quadrature must integrate SOS feature products exactly")
        self.feature_parity = np.concatenate([
            np.zeros(1, dtype=np.int64),
            np.concatenate([np.full(2 * ell + 1, ell % 2, dtype=np.int64)
                            for ell in range(1, self.degree // 2 + 1)])
            if self.degree else np.zeros(0, dtype=np.int64),
        ])
        rows, cols = np.tril_indices(self.n_features)
        rows, cols = rows[1:], cols[1:]
        if self.antipodal:
            same = self.feature_parity[rows] == self.feature_parity[cols]
            rows, cols = rows[same], cols[same]
        self.rows, self.cols = rows, cols
        self.n_parameters = len(self.rows)
        self.n_view_parameters = self.n_parameters
        self._moment_features = np.column_stack((
            np.ones(grid.Ns),
            np.sqrt(4 * np.pi)
            * real_spherical_harmonics(grid.xyz, self.degree),
        ))

    def _beta(self, beta):
        beta = np.asarray(beta, dtype=np.float64)
        if (beta.shape != (self.n_parameters,)
                or not np.all(np.isfinite(beta))):
            raise ValueError(
                "beta must be a finite vector of SOS coefficients")
        return beta

    def _components(self, beta):
        matrix = np.eye(self.n_features)
        matrix[self.rows, self.cols] += self._beta(beta)
        with np.errstate(over="ignore", invalid="ignore"):
            values = self.H @ matrix.T
            power = np.sum(values**2, axis=1)
            trace = float(np.sum(matrix**2))
            normalization = float(self.q @ power)
        if (not np.all(np.isfinite(power)) or not np.isfinite(trace)
                or not np.isfinite(normalization) or normalization <= 0):
            raise ValueError("SOS density overflow or invalid normalization")
        return values, power, trace, normalization

    def density(self, beta, directions=None):
        """Evaluate the SOS density relative to uniform sphere measure.

        ``directions`` contains unit vectors; omitting it evaluates the
        quadrature nodes. Divide by ``4*pi`` for density per steradian.
        """

        _, power, trace, _ = self._components(beta)
        if directions is not None:
            harmonics = real_spherical_harmonics(directions, self.degree // 2)
            features = np.column_stack((
                np.ones(len(harmonics)), np.sqrt(4 * np.pi) * harmonics,
            ))
            matrix = np.eye(self.n_features)
            matrix[self.rows, self.cols] += self._beta(beta)
            with np.errstate(over="ignore", invalid="ignore"):
                power = np.sum((features @ matrix.T)**2, axis=1)
            if not np.all(np.isfinite(power)):
                raise ValueError("SOS density overflow at requested directions")
        return power / trace

    def pmf(self, beta):
        """Return normalized masses including the sphere quadrature weights."""

        _, power, _, normalization = self._components(beta)
        return self.q * power / normalization

    def vjp(self, beta, grad_p):
        """Apply the transpose derivative of :meth:`pmf`."""

        grad_p = np.asarray(grad_p, dtype=np.float64)
        if grad_p.shape != self.q.shape or not np.all(np.isfinite(grad_p)):
            raise ValueError(
                "grad_p must be finite with one value per sphere node")
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
            viewing_representation="full real SH sum-of-squares density",
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
            viewing_haar_normalization=1.0,
            viewing_quadrature_normalization=normalization / trace,
            viewing_quadrature_gram_error=self.gram_error,
            viewing_mass_sum=float(p.sum()),
            viewing_min_mass=float(p.min()),
            viewing_max_mass=float(p.max()),
            viewing_ess=float(1.0 / np.sum(p * p)),
            viewing_sh_moments=moments.tolist(),
            viewing_sh_power_by_degree=moment_power,
            viewing_antipodal_symmetry=self.antipodal,
            viewing_moment_note=(
                "quadrature SH moments; exact coefficients require degree-2L "
                "quadrature"),
        )


def _single_order_weights(targets, lambdas):
    if len(targets) != 3 or len(lambdas) != 3:
        raise ValueError("targets and lambdas must each contain M1, M2, M3")
    weights = []
    for order, (target, value) in enumerate(zip(targets, lambdas), 1):
        weight = np.asarray(value, dtype=float)
        if (weight.shape not in ((), np.shape(target))
                or not np.all(np.isfinite(weight)) or np.any(weight < 0)):
            raise ValueError(
                f"M{order} weight must be finite, nonnegative, and scalar "
                "or target-shaped")
        weights.append(weight)
    active = [order for order, weight in enumerate(weights, 1)
              if np.any(weight > 0)]
    if len(active) != 1:
        raise ValueError("exactly one moment must have a positive weight")
    if not np.all(np.isfinite(targets[active[0] - 1])):
        raise ValueError("the selected moment target must be finite")
    return weights, active[0]


def _check_gradients(objective, x, ntheta):
    rng = np.random.default_rng(1729)
    probe = x.copy()
    probe[ntheta:] += 0.1 * rng.normal(size=len(probe) - ntheta)
    _, gradient = objective(probe)
    checks = {}
    for name, block in (("volume", slice(0, ntheta)),
                        ("viewing", slice(ntheta, None))):
        width = len(probe[block])
        if not width:
            checks[name] = dict(
                skipped=True,
                reason="fixed uniform degree-zero density",
            )
            continue
        delta = np.zeros_like(probe)
        direction = rng.normal(size=width)
        delta[block] = direction / np.linalg.norm(direction)
        step = 1e-5 * max(1.0, np.linalg.norm(probe[block]))
        numerical = (objective(probe + step * delta)[0]
                     - objective(probe - step * delta)[0]) / (2 * step)
        analytical = float(gradient @ delta)
        error = float(abs(numerical - analytical)
                      / max(1.0, abs(numerical), abs(analytical)))
        if not np.isfinite(error) or error > 2e-6:
            raise RuntimeError(
                f"low-degree {name} gradient check failed: "
                f"scaled error={error}")
        checks[name] = dict(
            numerical=float(numerical), analytical=analytical,
            scaled_error=error, step=float(step), skipped=False,
        )
    return checks


def fit_single_moment_low_degree(
        param, grid, targets, lambdas, theta0, beta0=None,
        degree=2, maxiter=5000, print_every=100,
        reflection_augmented=False):
    """Fit exactly one moment order over volume and viewing parameters."""

    targets = tuple(targets)
    weights, order = _single_order_weights(targets, lambdas)
    for name, value, lower in (("maxiter", maxiter, 1),
                               ("print_every", print_every, 0)):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or not np.isfinite(value) or int(value) != value
                or value < lower):
            raise ValueError(f"{name} must be an integer >= {lower}")
    density = LowDegreeViewingDensity(grid, degree=degree,
                                      antipodal=bool(reflection_augmented))
    theta0 = np.asarray(theta0, dtype=np.float64)
    ntheta = param.n_params
    if theta0.shape != (ntheta,) or not np.all(np.isfinite(theta0)):
        raise ValueError(
            "theta0 must be a finite vector with param.n_params entries")
    if beta0 is None:
        beta0 = np.zeros(density.n_parameters)
    x0 = np.r_[theta0, density._beta(beta0)]
    contrast_scales = np.ones(3)
    target_norm = max(
        float(np.linalg.norm(np.ravel(targets[order - 1]))), 1e-30)
    state = dict(nit=0, nfev=0)

    def objective(x):
        p = density.pmf(x[ntheta:])
        value, gtheta, gp, info = joint_objective_and_grad(
            x[:ntheta], p, param, grid, *targets,
            lambdas=weights,
            contrast_scales=contrast_scales, return_info=True,
            reflection_augmented=reflection_augmented,
        )
        gradient = np.r_[gtheta, density.vjp(x[ntheta:], gp)]
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "nonfinite low-degree moment objective or gradient")
        state.update(
            f=value,
            gnorm=float(np.linalg.norm(gradient)),
            relative_error=float(info[f"misfit{order}"] / target_norm),
        )
        state["nfev"] += 1
        return value, gradient

    gradient_checks = _check_gradients(objective, x0, ntheta)
    state.update(nit=0, nfev=0)

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
        ftol=1e-12, gtol=1e-10, maxcor=100,
    )
    result = minimize(
        objective, x0, jac=True, method="L-BFGS-B", callback=callback,
        options=options,
    )
    x = np.asarray(result.x, dtype=float).copy()
    if not np.isfinite(result.fun) or not np.all(np.isfinite(x)):
        raise FloatingPointError("nonfinite low-degree optimizer result")
    theta, beta = x[:ntheta].copy(), x[ntheta:].copy()
    p = density.pmf(beta)
    diagnostics = density.evaluate(beta)
    _, _, _, info = joint_objective_and_grad(
        theta, p, param, grid, *targets, lambdas=weights,
        contrast_scales=contrast_scales,
        return_info=True,
    )
    relative_errors = [float("nan")] * 3
    relative_errors[order - 1] = float(
        info[f"misfit{order}"] / target_norm)
    stage = dict(
        name=f"m{order}", theta=theta.copy(), beta=beta.copy(), p=p.copy(),
        contrast_scales=contrast_scales.copy(),
        rel_misfit=tuple(relative_errors), res=result,
        status=int(result.status), nit=int(result.nit), nfev=int(result.nfev),
        message=str(result.message), viewing_diagnostics=diagnostics,
    )
    if print_every:
        print(
            f"  [m{order} SOS{density.degree}] f={result.fun:.6e} "
            f"nit={result.nit:5d} nfev={result.nfev:5d} "
            f"{result.message}",
            flush=True,
        )
    return dict(
        theta=theta, beta=beta, p=p, x=x, stages=[stage], res=result,
        contrast_scales=contrast_scales, moment_order=order,
        viewing_degree=density.degree,
        n_view_parameters=density.n_parameters,
        viewing_diagnostics=diagnostics,
        gradient_checks=gradient_checks,
        nit=int(result.nit), nfev=int(result.nfev),
        status=int(result.status), message=str(result.message),
        optimizer_options=options,
    )
