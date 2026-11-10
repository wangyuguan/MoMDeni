"""Fitting the moments with the volume, the viewing density and the shift distribution, one moment order at a time.

Each stage (M1, M2, M3) minimizes the relative misfit of one moment over all three with L-BFGS-B (:func:`fit_moment`) from several starts
and keeps the best fit. M1 and M2 use ``N_INIT`` random starts (M2 start i begins at M1 fit i); the M2 solution whose model best explains
M3 (:func:`select_m2_start`) is then fitted to M3. The map is the M3 fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.optimize import minimize

from . import utils
from .moments import load_prepared_moments
from .shift import ShiftLaw, ShiftRings, read_records
from .utils import ROOT, git_commit, save_mrc, sha256_array, sha256_file, write_json
from .viewing import LowDegreeViewingDensity, viewing_grid
from .volume import SphFBParam

DEFAULT_ELL_MAX = 7
DEFAULT_VIEWING_DEGREE = 2
# Fixed settings of the M1 -> M2 -> M3 fits: N_INIT random starts of M1 and M2 (M3 runs once, from the M2 solution with the smallest
# M3 residual), and the L-BFGS iteration limits of M1/M2 and of M3.
N_INIT = 10
MAXITER = 10_000
M3_MAXITER = 500


@dataclass(frozen=True)
class ReconstructionConfig:
    """Configuration of one stage (the fit of one moment order)."""

    prepared_dir: Path
    output_dir: Path
    order: int
    ell_max: int = DEFAULT_ELL_MAX
    viewing_degree: int = DEFAULT_VIEWING_DEGREE
    symmetry: str = "C1"
    n_init: int = N_INIT
    maxiter: int = MAXITER
    device: str = "gpu"
    print_every: int = 100
    seed: int = 0
    initial_fit: Path | None = None
    resume: bool = False


def _since(start):
    return time.monotonic() - start


def fb_objective_and_grad(theta, p, param, mbar1, mbar2, mbar3, lambdas, return_info=False, shift=None):
    """Fourier--Bessel M1--M3 loss and its gradients with respect to the volume coefficients and the viewing masses.

    ``param`` must carry an ``fb_basis``; targets are the flat FB moments, the real parts of the
    empirical moments (equal to augmenting every image with its mirror). The model is symmetrized the
    same way: projections at antipodal viewing directions of a real volume are mirror images, so taking
    the real part of the predictions equals antipodally symmetrizing the viewing density, and the
    residuals compare real parts on both sides.

    ``shift = (rings, d)`` (one active order): the model moment of that order is the ``d``-mixture of its moments shifted by the rings of
    ``rings`` (:class:`shift.ShiftRings`); ``info["F"]`` then holds the per-ring moments.
    """
    basis = param.fb_basis
    targets = (mbar1, mbar2, mbar3)
    weights = [np.asarray(weight, dtype=float) for weight in lambdas]
    p = np.asarray(p, dtype=float)
    active = [bool(np.any(wt > 0)) for wt in weights]
    C = np.asarray(param.forward_images(theta))
    orders = tuple(k + 1 for k in range(3) if active[k])
    F = None
    if shift is None:
        predictions = basis.moments(C, p, orders=orders)
    else:
        rings, d = shift
        order = orders[0]
        if order == 3:      # M3 is formed image by image from the shifted images (shift.ShiftRings.m3_rings)
            predictions = list(basis.moments(C, p, orders=()))
            F = rings.m3_rings(C, p)
            predictions[2] = np.asarray(d) @ F
        else:
            predictions = list(basis.moments(C, p, orders=orders))
            predictions[order - 1], F = rings.predict(order, predictions[order - 1], d)
    masked, residuals, clean = [None] * 3, [None] * 3, [None] * 3
    costs, misfits = np.zeros(3), np.full(3, np.nan)
    for k in range(3):
        if not active[k]:
            continue
        # The residual is real, so the conjugate-linear gradient contraction below yields exactly d sum w (Re pred - t)^2.
        masked[k] = predictions[k].real.astype(np.complex128)
        residuals[k] = masked[k] - np.asarray(targets[k])
        misfits[k] = np.linalg.norm(residuals[k])
        costs[k] = float(np.sum(weights[k] * np.abs(residuals[k]) ** 2))
        clean[k] = weights[k] * residuals[k]
    if shift is None:
        G, grad_p = basis.moment_gradients(C, p, clean)
    elif order == 3:
        G3, g3 = rings.m3_gradients(C, p, d, clean[2])
        G, grad_p = basis.moment_gradients(C, p, [clean[0], clean[1], None])
        G, grad_p = G + G3, grad_p + g3
    else:
        clean[order - 1] = rings.pullback(d, clean[order - 1])
        G, grad_p = basis.moment_gradients(C, p, clean)
    grad_theta = param.adjoint_images(G)
    f = float(costs.sum())
    if not return_info:
        return f, grad_theta, grad_p
    info = {"f": f, "F": F}
    for k in range(3):
        info.update({f"f{k+1}": float(costs[k]), f"m{k+1}": predictions[k],
                     f"masked_m{k+1}": masked[k],
                     f"misfit{k+1}": float(misfits[k])})
    return f, grad_theta, grad_p, info


def fit_moment(law, param, grid, targets, lambdas, theta0, beta0, degree, maxiter, print_every):
    """Fit the single moment order with positive weight in ``lambdas`` over the volume, the viewing density and the ring weights of ``law``.

    One L-BFGS-B run from ``theta0``, ``beta0`` and the starting weights of ``law`` (:meth:`shift.ShiftLaw.start`), with d = u / sum(u), u >= 0;
    the fitted weights are recorded in ``law``. Returns the volume coefficients, the viewing parameters and masses, the optimizer result, the
    relative misfit of the moment and the viewing-density diagnostics."""
    weights = [np.asarray(value, dtype=float) for value in lambdas]
    order = next(order for order, weight in enumerate(weights, 1) if np.any(weight > 0))
    targets = tuple(targets)
    w3 = np.broadcast_to(weights[order - 1], np.shape(targets[order - 1]))
    rings = law.ring_operators(param.fb_basis)
    M = len(law.rho_px)
    basis = param.basis      # symmetric volume: theta = basis @ phi (None without symmetry)
    nt = param.n_params if basis is None else basis.shape[1]
    expand = (lambda phi: phi) if basis is None else (lambda phi: basis @ phi)
    reduce = (lambda g: g) if basis is None else (lambda g: basis.T @ g)
    density = LowDegreeViewingDensity(grid, degree=degree)
    b0 = np.asarray(np.zeros(density.n_parameters) if beta0 is None else beta0, dtype=np.float64)
    nb = len(b0)
    x0 = np.r_[reduce(np.asarray(theta0, float)), b0, law.start()]

    def objective(x):
        phi, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
        theta, total = expand(phi), u.sum()
        d = u / total
        value, g_theta, g_p, info = fb_objective_and_grad(
            theta, density.pmf(beta), param, *targets, lambdas=weights, return_info=True, shift=(rings, d))
        r = np.asarray(info[f"masked_m{order}"]) - np.asarray(targets[order - 1])
        g_d = 2 * np.real(((w3 * np.conj(r))[None, :] * info["F"].real.astype(complex)).sum(1))
        return value, np.r_[reduce(g_theta), density.vjp(beta, g_p), (g_d - d @ g_d) / total]

    state = dict(nit=0)

    def callback(_):
        state["nit"] += 1
        if state["nit"] % (print_every or 10 ** 9) == 0:
            print(f"    [m{order} shift law] it={state['nit']:5d}")

    maxiter = int(maxiter)
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", callback=callback,
                      bounds=[(None, None)] * (nt + nb) + [(0.0, None)] * M,
                      options=dict(maxiter=maxiter, maxfun=max(15000, 20 * maxiter), ftol=1e-11, gtol=1e-10, maxcor=100))
    x = np.asarray(result.x, float).copy()
    phi, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
    theta, d, p = expand(phi), u / u.sum(), density.pmf(x[nt:nt + nb])
    _, _, _, info = fb_objective_and_grad(theta, p, param, *targets, lambdas=weights, return_info=True, shift=(rings, d))
    norm = max(float(np.linalg.norm(np.ravel(targets[order - 1]))), 1e-30)
    record = law.record(d, result.fun, result.nit)
    print(f"  [m{order} shift law] loss={result.fun:.6e} nit={result.nit} mean radius {record['mean_radius_px']:.3f} px, "
          f"rms {record['rms_radius_px']:.3f} px")
    return dict(theta=theta.copy(), beta=beta.copy(), p=p.copy(), res=result,
                relative_error=float(info[f"misfit{order}"] / norm), viewing_diagnostics=density.evaluate(beta))


def deterministic_random_starts(n_parameters, n_init, seed=0):
    """Return reproducible Gaussian coefficient directions and provenance."""

    starts = []
    for index in range(n_init):
        child_seed = int(np.random.SeedSequence(
            [int(seed), 2, index]).generate_state(1)[0])
        direction = np.random.default_rng(child_seed).standard_normal(
            n_parameters)
        starts.append((
            index, child_seed, direction, sha256_array(direction)))
    return starts


def select_best_fit(records):
    """The record of minimum moment loss (the first one on ties)."""
    return min(records, key=lambda record: float(record["objective"]))


def _config_payload(
        config, prepared, param, density, grid, target_norm,
        warm_fit_paths=()):
    initial_fit = (None if config.initial_fit is None
                   else str(config.initial_fit.resolve(strict=True)))
    return dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        dataset=str(prepared.values["dataset"].reshape(())),
        order=config.order,
        fb_basis=prepared.fb_basis.summary(),
        objective=f"in-plane-invariant steerable FB M{config.order} only",
        prepared_directory=str(prepared.directory),
        prepared_sha256=prepared.report["prepared_sha256"],
        selected_images=prepared.n_images,
        moment_res=prepared.side,
        pixel_size_angstrom=prepared.pixel_size,
        native_side=int(prepared.values["native_side"].reshape(())),
        native_pixel_size_angstrom=float(
            prepared.values["native_pixel_size"].reshape(())),
        fb_coefficients=prepared.fb_basis.n_coefficients,
        n_m2_entries=int(np.prod(prepared.shapes[1])),
        canonical_m3_entries=int(np.prod(prepared.shapes[2])),
        ell_max=config.ell_max,
        n_volume_parameters=param.n_params,
        symmetry=config.symmetry,
        n_free_volume_parameters=param.n_params if param.basis is None else param.basis.shape[1],
        viewing_degree=config.viewing_degree,
        n_view_parameters=density.n_parameters,
        sphere_nodes=grid.Ns,
        target_norm=target_norm,
        target_sha256=sha256_array(prepared.target),
        n_initializations=config.n_init,
        maxiter=config.maxiter,
        initialization=(
            "deterministic Gaussian coefficient directions from "
            "SeedSequence([seed, 2, index]), rescaled to the "
            f"M{config.order} target norm"
            if not warm_fit_paths else
            f"M{config.order - 1} init_i volume and viewing density, with the volume analytically rescaled by least squares of order "
            f"{config.order} against M{config.order}"),
        initialization_seed=config.seed,
        initial_fit=initial_fit,
        initial_fit_files=[str(path) for path in warm_fit_paths],
        initial_fit_sha256=[sha256_file(path) for path in warm_fit_paths],
        kam_cryoem_commit=git_commit(ROOT),
    )


# Keys of config.json that must agree for a partially finished fit to continue (the others follow from them).
_RESUME_KEYS = (
    "order", "prepared_sha256", "ell_max", "viewing_degree", "symmetry",
    "n_initializations", "maxiter", "initialization_seed", "initial_fit_sha256",
)


def _read_completed_fit(output, index):
    """The record of initialization ``index`` if it finished before an interruption (its summary is written last)."""
    summary = output / f"init_{index}" / "summary.json"
    return json.loads(summary.read_text()) if summary.is_file() else None


def _copy_best_fit(output, record):
    source = output / f"init_{int(record['init_index'])}"
    shutil.copy2(source / "fit.npz", output / "fit.npz")
    shutil.copy2(source / "recon.mrc", output / "recon.mrc")


def _load_warm_start(path):
    """The path, volume coefficients and viewing parameters of one previous-order fit."""

    path = Path(path).resolve(strict=True)
    with np.load(path, allow_pickle=False) as saved:
        return path, np.asarray(saved["theta"], dtype=np.float64), np.asarray(saved["beta"], dtype=np.float64)


def _warm_fit_paths(initial_fit, n_init):
    """Resolve one previous-order fit per requested initialization."""

    directory = Path(initial_fit).resolve(strict=True)
    return [(directory / f"init_{index}" / "fit.npz").resolve(strict=True) for index in range(n_init)]


def _warm_starts(
        warm_fits, config, prepared, param, density, targets, lambdas):
    """Rescale each previous-order initialization for the next objective."""

    starts = []
    for index, (path, candidate, warm_beta) in enumerate(warm_fits):
        p = density.pmf(warm_beta)
        _, _, _, info = fb_objective_and_grad(
            candidate, p, param, *targets,
            lambdas=lambdas, return_info=True,
        )
        prediction = info[f"m{config.order}"]
        amplitude = float(np.vdot(prediction, prepared.target).real) / float(np.vdot(prediction, prediction).real)
        scale = float(np.sign(amplitude) * np.abs(amplitude) ** (1.0 / config.order))
        starts.append((
            index, -1, scale * candidate, warm_beta.copy(), scale,
            float(np.linalg.norm(prediction)), sha256_array(candidate),
            f"analytic M{config.order} rescaling of {path}",
        ))
    return starts


def reconstruct_order(config: ReconstructionConfig, shift: dict):
    """Run one stage with the ring shift distribution ``shift`` (``step_px``, ``max_px``), saved in ``shift_distribution.json``.

    Initialization i starts from the shift law of initialization i of ``config.initial_fit``. The laws of the initializations that finished
    before an interruption are kept (the finished initializations are a prefix) and the next fit is numbered after them."""
    path = config.output_dir / "shift_distribution.json"
    done = sum((config.output_dir / f"init_{i}" / "summary.json").is_file() for i in range(config.n_init))
    warm = [] if config.initial_fit is None else read_records(config.initial_fit / "shift_distribution.json")
    law = ShiftLaw(shift["step_px"], shift["max_px"], warm=[record["d"] for record in warm], finished=read_records(path)[:done], path=path)
    reconstruct_single_moment(config, law)


def reconstruct_single_moment(config: ReconstructionConfig, law: ShiftLaw):
    """Fit one moment order from every start (:func:`fit_moment`) and keep the start with minimum moment loss."""

    output = config.output_dir.resolve()
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not config.resume:
        raise FileExistsError(
            "use a new or empty reconstruction directory, or set resume=True")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    setup_started = time.monotonic()
    prepared = load_prepared_moments(config.prepared_dir, config.order)
    target_norm = float(np.linalg.norm(prepared.target))

    utils.use_device(config.device)
    # FB moments are already in-plane averaged: viewing directions only; the rule integrates the M3 model exactly.
    grid = viewing_grid(3 * config.ell_max + config.viewing_degree)
    param = SphFBParam(prepared.side, config.ell_max, grid.rots, fb_basis=prepared.fb_basis, symmetry=config.symmetry)
    density = LowDegreeViewingDensity(grid, degree=config.viewing_degree)
    targets = [np.zeros(shape, dtype=np.complex128)
               for shape in prepared.shapes]
    targets[config.order - 1] = prepared.target
    lambdas = [0.0, 0.0, 0.0]
    lambdas[config.order - 1] = 1.0 / target_norm**2
    beta_uniform = np.zeros(density.n_parameters)
    if config.initial_fit is None:
        p_uniform = density.pmf(beta_uniform)
        starts = []
        for index, child_seed, direction, direction_hash in \
                deterministic_random_starts(
                    param.n_params, config.n_init, config.seed):
            direction = param.project(direction)
            _, _, _, info = fb_objective_and_grad(
                direction, p_uniform, param, *targets,
                lambdas=lambdas, return_info=True,
            )
            prediction = info[f"m{config.order}"]
            prediction_norm = float(np.linalg.norm(prediction))
            scale = (target_norm / prediction_norm) ** (1.0 / config.order)
            starts.append((
                index, child_seed, scale * direction, beta_uniform.copy(),
                scale, prediction_norm, direction_hash,
                f"random Gaussian direction scaled to M{config.order} "
                "target norm",
            ))
        warm_fit_paths = []
    else:
        # M{order} start i is M{order-1} init_i.
        warm_fit_paths = _warm_fit_paths(config.initial_fit, config.n_init)
        warm_fits = [_load_warm_start(path) for path in warm_fit_paths]
        starts = _warm_starts(
            warm_fits, config, prepared, param, density, targets, lambdas)

    payload = _config_payload(
        config, prepared, param, density, grid, target_norm,
        warm_fit_paths=warm_fit_paths)
    payload["init_seeds"] = [start[1] for start in starts]
    payload["initialization_sources"] = [start[7] for start in starts]
    payload["m3_warm_start_semantics"] = (
        None if config.initial_fit is None else
        f"M{config.order} start i is M{config.order - 1} init_i with its fitted viewing density; each volume receives the analytic "
        f"order-{config.order} least-squares amplitude against "
        f"M{config.order}")
    config_path = output / "config.json"
    if nonempty:
        previous = json.loads(config_path.read_text())
        mismatched = {
            key: (previous.get(key), payload.get(key))
            for key in _RESUME_KEYS
            if previous.get(key) != payload.get(key)
        }
        if mismatched:
            raise ValueError(
                f"resume configuration differs from existing fit: {mismatched}")
        payload["created_utc"] = previous.get(
            "created_utc", payload["created_utc"])
    else:
        write_json(config_path, payload)

    completed = {}
    if config.resume:
        for index in range(config.n_init):
            record = _read_completed_fit(output, index)
            if record is not None:
                completed[index] = record
    setup_elapsed = _since(setup_started)
    fit_loop_started = time.monotonic()
    optimization_elapsed_this_run = 0.0
    completed_this_run = 0
    records = []
    for (index, child_seed, theta0, beta0, scale, prediction_norm,
         direction_hash, start_source) in starts:
        if index in completed:
            records.append(completed[index])
            print(
                f"M{config.order} start {index + 1}/{len(starts)}: "
                "using completed resume record",
                flush=True,
            )
            continue
        init_started = time.monotonic()
        init = output / f"init_{index}"
        init.mkdir(exist_ok=True)
        initial_loss = float(fb_objective_and_grad(
            theta0, density.pmf(beta0), param, *targets,
            lambdas=lambdas,
        )[0])
        np.savez(
            init / "start.npz",
            theta=theta0, beta=beta0, p=density.pmf(beta0),
            seed=child_seed, moment_order=config.order,
            scale=scale,
            prediction_norm_before_scaling=prediction_norm,
            target_norm=target_norm,
            direction_sha256=direction_hash,
            objective=initial_loss,
            source=start_source,
        )
        print(
            f"M{config.order} start {index + 1}/{len(starts)}: "
            f"initial normalized loss={initial_loss:.8e}",
            flush=True,
        )
        optimization_started = time.monotonic()
        result = fit_moment(law, param, grid, targets, lambdas, theta0, beta0, config.viewing_degree, config.maxiter,
                            config.print_every)
        optimization_elapsed = _since(optimization_started)
        optimization_elapsed_this_run += optimization_elapsed
        optimizer = result["res"]
        loss = float(optimizer.fun)
        relative_error = result["relative_error"]
        np.savez(
            init / "fit.npz",
            theta=result["theta"], beta=result["beta"], p=result["p"],
            init_index=index, initialization_source=start_source,
            objective=loss, normalized_active_moment_loss=loss,
            relative_moment_error=relative_error,
            moment_order=config.order,
            viewing_degree=config.viewing_degree,
            ell_max=config.ell_max,
            grid_rots=grid.rots, grid_xyz=grid.xyz, grid_q=grid.q,
            prepared_sha256=prepared.report["prepared_sha256"],
        )
        save_mrc(
            param.volume(result["theta"]).transpose(2, 1, 0),
            init / "recon.mrc", prepared.pixel_size,
        )
        record = dict(
            init_index=index,
            seed=child_seed,
            moment_order=config.order,
            initial_objective=initial_loss,
            initialization_source=start_source,
            objective=loss,
            normalized_active_moment_loss=loss,
            relative_moment_error=relative_error,
            nit=int(optimizer.nit),
            nfev=int(optimizer.nfev),
            status=int(optimizer.status),
            success=bool(optimizer.success),
            message=str(optimizer.message),
            gradient_norm=float(np.linalg.norm(optimizer.jac)),
            viewing=result["viewing_diagnostics"],
            optimization_elapsed_seconds=optimization_elapsed,
        )
        record["elapsed_seconds"] = _since(init_started)
        record["nonoptimization_elapsed_seconds"] = max(
            0.0, record["elapsed_seconds"] - optimization_elapsed)
        write_json(init / "summary.json", record)
        completed_this_run += 1
        records.append(record)
        best = select_best_fit(records)
        _copy_best_fit(output, best)
        partial = dict(payload)
        partial.update(
            fits=sorted(records, key=lambda item: item["init_index"]),
            best_init_by_loss=int(best["init_index"]),
            fit_complete=len(records) == len(starts),
            complete=False,
        )
        write_json(output / "summary.json", partial)
        print("Fit result: " + json.dumps(record), flush=True)

    fit_loop_elapsed = _since(fit_loop_started)
    final_output_started = time.monotonic()
    records.sort(key=lambda item: item["init_index"])
    best = select_best_fit(records)
    _copy_best_fit(output, best)
    with np.load(output / "fit.npz", allow_pickle=False) as saved:
        beta = np.asarray(saved["beta"], dtype=np.float64)
    _save_viewing_plot(output, density, beta)

    optimization_total = float(sum(record["optimization_elapsed_seconds"] for record in records))
    initializations_total = float(sum(record["elapsed_seconds"] for record in records))

    summary = dict(payload)
    summary.update(
        fits=records,
        best_init_by_loss=int(best["init_index"]),
        best_normalized_active_moment_loss=float(best["objective"]),
        fit_complete=True,
        completed_in_this_invocation=completed_this_run,
    )
    final_output_elapsed = _since(final_output_started)
    elapsed = _since(started)
    timings = dict(
        setup=setup_elapsed,
        initialization_fit_loop=fit_loop_elapsed,
        optimization_total=optimization_total,
        optimization_this_invocation=optimization_elapsed_this_run,
        initializations_total=initializations_total,
        final_output=final_output_elapsed,
        total=elapsed,
    )
    summary["timings_seconds"] = timings
    summary["elapsed_seconds"] = elapsed
    summary["complete"] = True
    write_json(output / "summary.json", summary)

    # Include one complete summary write in the top-level output and elapsed
    # timings.  The second JSON rewrite only updates the measured values.
    final_output_elapsed = _since(final_output_started)
    elapsed = _since(started)
    timings["final_output"] = final_output_elapsed
    timings["total"] = elapsed
    summary["timings_seconds"] = timings
    summary["elapsed_seconds"] = elapsed
    write_json(output / "summary.json", summary)


def _save_viewing_plot(output, density, beta):
    """Plot the fitted SH density on a dense sphere grid, in sr^-1."""

    longitude_edges = np.linspace(-np.pi, np.pi, 361)
    latitude_edges = np.linspace(-np.pi / 2, np.pi / 2, 181)
    longitude, latitude = np.meshgrid(
        (longitude_edges[:-1] + longitude_edges[1:]) / 2,
        (latitude_edges[:-1] + latitude_edges[1:]) / 2,
    )
    directions = np.column_stack((
        (np.cos(latitude) * np.cos(longitude)).ravel(),
        (np.cos(latitude) * np.sin(longitude)).ravel(),
        np.sin(latitude).ravel(),
    ))
    probability_density = density.density(beta, directions).reshape(
        latitude.shape) / (4 * np.pi)
    figure, axis = plt.subplots(
        subplot_kw={"projection": "mollweide"}, figsize=(6.5, 3.8))
    surface = axis.pcolormesh(
        longitude_edges, latitude_edges, probability_density,
        cmap="viridis", shading="flat", rasterized=True, vmin=0,
    )
    axis.set_longitude_grid(60)
    axis.set_latitude_grid(30)
    axis.grid(alpha=0.2)
    figure.colorbar(
        surface, ax=axis, label="Probability density (sr$^{-1}$)", shrink=0.75)
    figure.tight_layout()
    figure.savefig(output / "viewing_density.pdf")
    plt.close(figure)


def select_m2_start(prepared_dir, m2_dir, output_dir, *, ell_max, viewing_degree, symmetry, device):
    """Pick the M2 initialization whose solution best explains M3, before any M3 fitting, and write it alone to ``output_dir``.

    For every M2 initialization the shifted model M3 of its volume, viewing density and shift weights is formed and given its
    least-squares amplitude (as the M3 warm start rescales the volume); the initialization with the smallest relative residual
    ||a Re M3_model - M3||^2 / ||M3||^2 is chosen. ``output_dir`` then holds ``init_0`` (that fit), its shift law and
    ``selection.json``, the layout of an M2 stage with one initialization, so M3 starts from it."""
    started = time.monotonic()
    m2_dir, output_dir = Path(m2_dir), Path(output_dir)
    utils.use_device(device)
    prepared = load_prepared_moments(prepared_dir, 3)
    grid = viewing_grid(3 * ell_max + viewing_degree)
    param = SphFBParam(prepared.side, ell_max, grid.rots, fb_basis=prepared.fb_basis, symmetry=symmetry)
    density = LowDegreeViewingDensity(grid, degree=viewing_degree)
    law = json.loads((m2_dir / "shift_distribution.json").read_text())
    rings = ShiftRings(prepared.fb_basis, np.asarray(law["rho_px"]))
    target = np.real(prepared.target)
    fits = sorted(json.loads((m2_dir / "summary.json").read_text())["fits"], key=lambda fit: fit["init_index"])
    losses = []
    for fit in fits:
        index = int(fit["init_index"])
        with np.load(m2_dir / f"init_{index}" / "fit.npz", allow_pickle=False) as saved:
            theta, beta = np.asarray(saved["theta"]), np.asarray(saved["beta"])
        d = np.asarray(law["initializations"][index]["d"])
        model = np.real(d @ rings.m3_rings(np.asarray(param.forward_images(theta)), density.pmf(beta)))
        amplitude = float(model @ target / (model @ model))
        losses.append(float(np.sum((amplitude * model - target) ** 2) / np.sum(target ** 2)))
    best = int(fits[int(np.argmin(losses))]["init_index"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "init_0").mkdir()
    for name in ("fit.npz", "recon.mrc"):
        shutil.copy2(m2_dir / f"init_{best}" / name, output_dir / "init_0" / name)
    (output_dir / "shift_distribution.json").write_text(json.dumps(
        dict(rho_px=law["rho_px"], initializations=[law["initializations"][best]]), indent=2))
    elapsed = time.monotonic() - started
    record = dict(
        method="M3 residual of every M2 solution (shifted model M3 with its least-squares amplitude), before M3 fitting",
        selected_m2_init=best,
        m3_losses={int(fit["init_index"]): loss for fit, loss in zip(fits, losses)},
        m2_losses={int(fit["init_index"]): float(fit["objective"]) for fit in fits},
        m2_dir=str(m2_dir), elapsed_seconds=elapsed, timings_seconds=dict(total=elapsed))
    (output_dir / "selection.json").write_text(json.dumps(record, indent=2))
    print("[M2 selection] M3 losses of the M2 solutions: " + ", ".join(f"init_{int(f['init_index'])} {loss:.4e}" for f, loss in zip(fits, losses))
          + f"; selected init_{best}", flush=True)
    return best
