"""Population shift distribution for the Fourier-Bessel moments.

The images are only centered by the guarded common shift; the remaining per-image shifts are not estimated but modeled by an isotropic radial
law on a ring grid,

    p(s) = sum_j d_j delta(|s| - rho_j) / (2 pi rho_j),    d_j >= 0,  sum_j d_j = 1,

with rho_j in pixels of the moment grid (equispaced, sub-pixel allowed). For a ring the shift-angle average is analytic and the shifted moments
are Kronecker products of the real mixing matrices

    K_j[(q,a),(p,u)] = sum_k U_|q|[k,a] U_|p|[k,u] J_{q-p}(2 pi k rho_j)

acting on the signed coefficient tensors of the unshifted moments x_k(A, B), so no radial integral is left:

    y_k = sum_j d_j S_{k,j} x_k(A, B).

The weights d are fitted jointly with the volume and the viewing density by one L-BFGS-B run, d = u / sum(u) with u >= 0 (box constraints keep
d on the simplex exactly; u = e_0 is "no shift"). Only the M3 stage uses the shift law. `shift_law` temporarily wraps the moment evaluation and
the single-moment fit, so none of the reconstruction modules that define the implementation fingerprint is modified.
"""
import contextlib
import inspect
import json

import numpy as np
from scipy.special import jv

from . import moments as _moments_module
from . import postprocessing as _postprocessing
from . import viewing as _viewing

DEFAULT_STEP_PX = 0.25
DEFAULT_MAX_PX = 5.0


def ring_radii(step_px, max_px):
    """Equispaced ring radii 0, step, 2 step, ... <= max (moment-grid pixels)."""
    if not (step_px > 0 and max_px >= step_px):
        raise ValueError("need 0 < step <= max for the shift rings")
    return np.arange(0.0, max_px + 1e-9, step_px)


class ShiftRings:
    """Ring operators S_{k,j} on the stored (flat) FB moments of an ``FBMomentBasis``."""

    def __init__(self, basis, rho, device=None):
        import torch
        self.torch = torch
        self.basis = basis
        self.L = basis.ell_max
        self.T = [int(t) for t in basis.counts]
        self.rho = np.asarray(rho, float)
        self.m = len(self.rho)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.orders = list(range(-self.L, self.L + 1))
        self.off, n = {}, 0
        for p in self.orders:
            self.off[p] = (n, n + self.T[abs(p)])
            n += self.T[abs(p)]
        self.n = n
        nodes, radial = basis.nodes, basis.radial
        K = np.zeros((self.m, n, n))
        for j, r in enumerate(self.rho):
            for q in self.orders:
                for p in self.orders:
                    J = jv(q - p, 2 * np.pi * nodes * r)
                    K[j, self.off[q][0]:self.off[q][1], self.off[p][0]:self.off[p][1]] = (
                        radial[abs(q)].T @ (J[:, None] * radial[abs(p)]))
        self.K = torch.tensor(K, dtype=torch.float64, device=self.device)
        self._m3_maps()

    # ---- stored <-> signed full tensors ---------------------------------- #
    def _m3_maps(self):
        """The stored block (a,b,c) is E[a_a a_b conj a_c] = (-1)^c X3[(a,b,-c)]. The signed tensor X3 also holds every axis permutation
        and the reflected triple (-a,-b,c) = conj(X3[(a,b,-c)])."""
        torch = self.torch
        idx = lambda p: torch.arange(*self.off[p], device=self.device)
        self.m3 = []
        for blk in self.basis.m3_blocks:
            a, b, c = blk["q"]
            self.m3.append((blk["slice"], float(np.sqrt(blk["weight"])), (-1.0) ** c,
                            [idx(a), idx(b), idx(-c)], [idx(-a), idx(-b), idx(c)],
                            (len(idx(a)), len(idx(b)), len(idx(-c)))))

    @staticmethod
    def _grid(ix):
        return ix[0][:, None, None], ix[1][None, :, None], ix[2][None, None, :]

    def _full3(self, m3):
        torch = self.torch
        X = torch.zeros((self.n,) * 3, dtype=torch.complex128, device=self.device)
        for sl, sw, sign, ax, rf, shape in self.m3:
            blk = (sign / sw) * m3[sl].reshape(shape)
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                X = X.index_put(self._grid([ax[t] for t in perm]), blk.permute(*perm))
                X = X.index_put(self._grid([rf[t] for t in perm]), torch.conj(blk).permute(*perm))
        return X

    def _read3(self, Y):
        torch = self.torch
        return torch.cat([(sw * sign * Y[self._grid(ax)]).reshape(-1)
                          for sl, sw, sign, ax, rf, shape in self.m3])

    # ---- operators --------------------------------------------------------- #
    def apply(self, order, m0):
        """``F[j]``: the ring-j moment of the flat stored moment ``m0`` (complex tensor on ``self.device``)."""
        torch = self.torch
        K = self.K.to(torch.complex128)
        if order == 1:
            s0 = slice(*self.off[0])
            return torch.einsum("jab,b->ja", K[:, s0, s0], m0)
        if order == 2:
            out = []
            for blk in self.basis.m2_blocks:
                sq, acc = slice(*self.off[blk["q"]]), 0
                for p in self.orders:
                    pb = self.basis.m2_blocks[abs(p)]
                    M = m0[pb["slice"]].reshape(self.T[abs(p)], self.T[abs(p)]) / np.sqrt(pb["weight"])
                    M = M if p >= 0 else M.T
                    Kq = K[:, sq, slice(*self.off[p])]
                    acc = acc + torch.einsum("jab,bc,jdc->jad", Kq, M, Kq)
                out.append((np.sqrt(blk["weight"]) * acc).reshape(self.m, -1))
            return torch.cat(out, dim=1)
        if order == 3:
            Y = torch.einsum("jia,jkb,jlc,abc->jikl", K, K, K, self._full3(m0))
            return torch.stack([self._read3(Y[j]) for j in range(self.m)])
        raise ValueError(f"unsupported moment order {order}")

    def predict(self, order, m0, d):
        """``d @ F`` for numpy ``m0`` and ``d``; also returns ``F`` as numpy."""
        torch = self.torch
        F = self.apply(order, torch.tensor(np.asarray(m0), dtype=torch.complex128, device=self.device))
        y = torch.tensor(np.asarray(d), dtype=torch.complex128, device=self.device) @ F
        return y.cpu().numpy(), F.cpu().numpy()

    def pullback(self, order, m0, d, clean):
        """Adjoint for the real inner product: gradient of ``sum <clean, d @ F(m0)>`` (real and imaginary parts) with respect to ``m0``."""
        torch = self.torch
        mr = torch.tensor(np.stack([m0.real, m0.imag], -1), dtype=torch.float64, device=self.device, requires_grad=True)
        y = torch.tensor(np.asarray(d), dtype=torch.complex128, device=self.device) @ self.apply(order, torch.view_as_complex(mr))
        c = torch.tensor(np.asarray(clean), dtype=torch.complex128, device=self.device)
        (g,) = torch.autograd.grad((torch.view_as_real(y) * torch.view_as_real(c)).sum(), mr)
        return (g[..., 0] + 1j * g[..., 1]).detach().cpu().numpy()


class _Law:
    """State shared by the wrapped moment functions and the joint fit."""

    def __init__(self, step_px, max_px):
        self.rho_px = ring_radii(step_px, max_px)
        self.rings = None
        self.active = False
        self.d = None
        self.m0 = None
        self.F = None
        self.records = []


@contextlib.contextmanager
def shift_law(step_px=DEFAULT_STEP_PX, max_px=DEFAULT_MAX_PX):
    """Fit the shift law jointly with volume and viewing density in the M3 stage while the context is open.

    Yields the state object; its ``records`` list has one entry per initialization (ring radii in pixels, fitted weights, losses)."""
    law = _Law(step_px, max_px)
    FB = _moments_module.FBMomentBasis
    orig_moments, orig_gradients = FB.moments, FB.moment_gradients
    orig_fit = _postprocessing.fit_single_moment_low_degree

    def moments(self, C, w, orders=(1, 2, 3), batch_size=4096):
        out = list(orig_moments(self, C, w, orders=orders, batch_size=batch_size))
        if law.active and out[2] is not None:
            law.m0 = out[2]
            out[2], law.F = law.rings.predict(3, out[2], law.d)
        return tuple(out)

    def gradients(self, C, w, clean):
        if law.active and clean[2] is not None:
            clean = list(clean)
            clean[2] = law.rings.pullback(3, law.m0, law.d, clean[2])
        return orig_gradients(self, C, w, clean)

    def fit(param, grid, targets, lambdas, *args, **kwargs):
        weights, order = _viewing._single_order_weights(tuple(targets), lambdas)
        if order != 3:
            return orig_fit(param, grid, targets, lambdas, *args, **kwargs)
        bound = inspect.signature(orig_fit).bind(param, grid, targets, lambdas, *args, **kwargs)
        bound.apply_defaults()
        result = _joint_fit(law, orig_fit, param, grid, targets, lambdas, weights[2], bound.arguments)
        law.active = False
        return result

    FB.moments, FB.moment_gradients = moments, gradients
    _postprocessing.fit_single_moment_low_degree = fit
    try:
        yield law
    finally:
        FB.moments, FB.moment_gradients = orig_moments, orig_gradients
        _postprocessing.fit_single_moment_low_degree = orig_fit


def _joint_fit(law, orig_fit, param, grid, targets, lambdas, weight, arguments, log=print):
    from scipy.optimize import minimize
    from .optimization import joint_objective_and_grad

    refl = bool(arguments.get("reflection_augmented", False))
    degree, maxiter = arguments.get("degree", 2), int(arguments.get("maxiter", 5000))
    weights = _viewing._single_order_weights(tuple(targets), lambdas)[0]
    w3 = np.broadcast_to(np.asarray(weight, float), np.shape(targets[2]))
    if law.rings is None:
        law.rings = ShiftRings(param.fb_basis, law.rho_px)
    M, nt = len(law.rho_px), param.n_params
    # a one-iteration call of the unchanged fit provides the result structure
    template = orig_fit(param, grid, targets, lambdas, arguments["theta0"], beta0=arguments.get("beta0"), degree=degree,
                        maxiter=1, print_every=0, reflection_augmented=refl)
    density = _viewing.LowDegreeViewingDensity(grid, degree=degree, antipodal=refl)
    beta0 = arguments.get("beta0")
    b0 = density._beta(np.zeros(density.n_parameters) if beta0 is None else beta0)
    nb = len(b0)
    u0 = np.zeros(M)
    u0[0] = 1.0
    x0 = np.r_[np.asarray(arguments["theta0"], float), b0, u0]
    scales = np.ones(3)

    def objective(x):
        theta, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
        total = u.sum()
        law.active, law.d = True, u / total
        value, g_theta, g_p, info = joint_objective_and_grad(
            theta, density.pmf(beta), param, grid, *targets, lambdas=weights, contrast_scales=scales,
            return_info=True, reflection_augmented=refl)
        r = np.asarray(info["masked_m3"]) - np.asarray(targets[2])
        F = law.F.real.astype(complex) if refl else law.F
        g_d = 2 * np.real(((w3 * np.conj(r))[None, :] * F).sum(1))
        return value, np.r_[g_theta, density.vjp(beta, g_p), (g_d - law.d @ g_d) / total]

    state = dict(nit=0)

    def callback(_):
        state["nit"] += 1
        if state["nit"] % int(arguments.get("print_every") or 0 or 10 ** 9) == 0:
            log(f"    [m3 shift law] it={state['nit']:5d}")

    result = minimize(objective, x0, jac=True, method="L-BFGS-B", callback=callback,
                      bounds=[(None, None)] * (nt + nb) + [(0.0, None)] * M,
                      options=dict(maxiter=maxiter, maxfun=max(15000, 20 * maxiter), ftol=1e-10, gtol=1e-10, maxcor=100))
    x = np.asarray(result.x, float).copy()
    theta, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
    d, p = u / u.sum(), density.pmf(x[nt:nt + nb])
    # same final evaluation as fit_single_moment_low_degree (the caller checks loss == rel_misfit**2)
    law.active, law.d = True, d
    _, _, _, info = joint_objective_and_grad(theta, p, param, grid, *targets, lambdas=weights, contrast_scales=scales, return_info=True)
    norm = max(float(np.linalg.norm(np.ravel(targets[2]))), 1e-30)
    stage = dict(template["stages"][0])
    stage["rel_misfit"] = (float("nan"), float("nan"), float(info["misfit3"] / norm))
    update = dict(theta=theta.copy(), beta=beta.copy(), p=p.copy(), res=result, status=int(result.status), nit=int(result.nit),
                  nfev=int(result.nfev), message=str(result.message), viewing_diagnostics=density.evaluate(beta))
    out = dict(template)
    for key, value in update.items():
        out[key] = value
        stage[key] = value
    out.update(x=np.r_[theta, beta], stages=[stage],
               shift_law=dict(rho_px=law.rho_px.tolist(), d=d.tolist()))
    law.records.append(dict(rho_px=law.rho_px.tolist(), d=d.tolist(), loss=float(result.fun), nit=int(result.nit),
                            mean_radius_px=float((d * law.rho_px).sum()),
                            rms_radius_px=float(np.sqrt((d * law.rho_px ** 2).sum()))))
    log(f"  [m3 shift law] loss={result.fun:.6e} nit={result.nit} mean radius {law.records[-1]['mean_radius_px']:.3f} px, "
        f"rms {law.records[-1]['rms_radius_px']:.3f} px")
    return out


def write_records(law, path):
    """Save the fitted laws (one per initialization, in fit order)."""
    with open(path, "w") as f:
        json.dump(dict(model="isotropic radial law on rings; d >= 0, sum d = 1", rho_px=law.rho_px.tolist(),
                       initializations=law.records), f, indent=2)
