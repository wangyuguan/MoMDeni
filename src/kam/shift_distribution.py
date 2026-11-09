"""Population shift distribution for the Fourier-Bessel moments.

The images are not shifted; their per-image shifts are not estimated but modeled by an isotropic radial law on a ring grid,

    p(s) = sum_j d_j delta(|s| - rho_j) / (2 pi rho_j),    d_j >= 0,  sum_j d_j = 1,

with rho_j in pixels of the moment grid (equispaced, sub-pixel allowed). For a ring the shift-angle average is analytic and the shifted moments
are Kronecker products of the real mixing matrices

    K_j[(q,a),(p,u)] = sum_k U_|q|[k,a] U_|p|[k,u] J_{q-p}(2 pi k rho_j)

acting on the signed coefficient tensors of the unshifted moments x_k(A, B), so no radial integral is left:

    y_k = sum_j d_j S_{k,j} x_k(A, B).

M1 and M2 are evaluated this way. M3 is evaluated image by image instead, which gives the same moment without the dense signed
(n x n x n) M3 tensor: on a stored block (a, b, a+b) the shift angle cancels, so the shifted moment of model image c is the in-plane
average of the triple products of s = K_j R_alpha c (R_alpha multiplies order q by e^{i q alpha}), and that average is exact with
A = 3L + 1 equispaced angles (the phases e^{i (p1 + p2 - p3) alpha} have |p1 + p2 - p3| <= 3L). The stored M3 blocks are then summed over
the rows s, with masses d_j w_i / A, by the same block formulas as `FBMomentBasis.moments` and `moment_gradients`.

The weights d are fitted jointly with the volume and the viewing density by one L-BFGS-B run, d = u / sum(u) with u >= 0 (box constraints keep
d on the simplex exactly; u = e_0 is "no shift"). Every stage (M1, M2, M3) fits its own d; M2 initialization i starts from the d of M1
initialization i and M3 initialization i from the d of M2 initialization i. `shift_law` temporarily wraps the moment evaluation
and the single-moment fit.
"""
import contextlib
import json
from pathlib import Path

import numpy as np
from scipy.special import jv

from . import moments as _moments_module
from . import postprocessing as _postprocessing
from . import viewing as _viewing


def ring_radii(step_px, max_px):
    """Equispaced ring radii 0, step, 2 step, ... <= max (moment-grid pixels)."""
    return np.arange(0.0, max_px + 1e-9, step_px)


class ShiftRings:
    """Ring operators S_{k,j} on the stored (flat) FB moments of an ``FBMomentBasis``."""

    def __init__(self, basis, rho):
        import torch
        self.torch = torch
        self.basis = basis
        self.L = basis.ell_max
        self.T = [int(t) for t in basis.counts]
        self.rho = np.asarray(rho, float)
        self.m = len(self.rho)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self._m3_rows()
        self._m2_maps()

    # ---- M3 row operator ----------------------------------------------------- #
    def _m3_rows(self):
        """Operators for the shifted images s = K_j R_alpha c: the rows of K_j for the stored orders 0..L, the in-plane phases of the
        A = 3L + 1 angles on the signed orders, and the M3 blocks. Rings are processed in chunks of about 2 GB of block intermediates."""
        torch = self.torch
        rows = np.concatenate([np.arange(*self.off[q]) for q in range(self.L + 1)])
        self._K3 = self.K[:, torch.as_tensor(rows, device=self.device)].to(torch.complex128)        # (rings, sum_q T_q, n)
        self.n_angles = 3 * self.L + 1
        alpha = 2 * np.pi * np.arange(self.n_angles) / self.n_angles
        signed = np.concatenate([np.full(self.T[abs(p)], p) for p in self.orders])
        self._phase = torch.as_tensor(np.exp(1j * alpha[:, None] * signed[None, :]), device=self.device)  # (A, n)
        self._signed = [(p, self.basis.slices[abs(p)]) for p in self.orders]
        edges = np.r_[0, np.cumsum(self.T)]
        self._stored = [slice(edges[q], edges[q + 1]) for q in range(self.L + 1)]
        self._m3_ring_chunk = None

    def _shifted_rows(self, C):
        """``S[j, alpha, i]``: the stored orders 0..L of K_j R_alpha c_i for the rows c_i of ``C`` (torch, (rings, A, N, sum_q T_q))."""
        torch = self.torch
        Ct = torch.as_tensor(np.ascontiguousarray(C, dtype=np.complex128), device=self.device)
        X = torch.cat([Ct[:, s] if p >= 0 else (-1.0) ** p * torch.conj(Ct[:, s]) for p, s in self._signed], dim=1)
        return torch.einsum("jqn,avn->javq", self._K3, self._phase[:, None, :] * X[None])

    def _ring_chunks(self, n_rows):
        if self._m3_ring_chunk is None:
            t = max(self.T)
            self._m3_ring_chunk = max(1, int(2e9 // (n_rows * t * t * 16)))
        return range(0, self.m, self._m3_ring_chunk)

    def m3_rings(self, C, w):
        """``F[j]``: the stored M3 of the images ``C`` with masses ``w`` shifted by ring j (numpy, (rings, M3 size))."""
        torch = self.torch
        S = self._shifted_rows(C)
        n_rows = self.n_angles * len(w)
        ww = torch.as_tensor(np.asarray(w, dtype=float) / self.n_angles, device=self.device).repeat(self.n_angles).to(torch.complex128)
        F = torch.empty((self.m, self.basis.m3_size), dtype=torch.complex128, device=self.device)
        for j0 in self._ring_chunks(n_rows):
            X = S[j0:j0 + self._m3_ring_chunk].reshape(-1, n_rows, S.shape[-1])                       # (ring chunk, A N, sum_q T_q)
            for block in self.basis.m3_blocks:
                a, b, c = block["q"]
                P = ((ww[:, None] * X[..., self._stored[a]])[..., :, None] * X[..., None, self._stored[b]]).reshape(len(X), n_rows, -1)
                F[j0:j0 + len(X), block["slice"]] = np.sqrt(block["weight"]) * (
                    P.transpose(1, 2) @ torch.conj(X[..., self._stored[c]])).reshape(len(X), -1)
        return F.cpu().numpy()

    def m3_gradients(self, C, w, d, clean):
        """Gradients of ``2 Re vdot(clean, d @ F(C, w))`` with respect to ``C`` (df = Re vdot(G, dC)) and ``w``, as
        `FBMomentBasis.moment_gradients` for the unshifted M3."""
        torch = self.torch
        S = self._shifted_rows(C)
        N, n_rows = len(w), self.n_angles * len(w)
        dt = torch.as_tensor(np.asarray(d, dtype=float), device=self.device)
        wt = torch.as_tensor(np.asarray(w, dtype=float), device=self.device)
        r3 = torch.as_tensor(np.asarray(clean, dtype=np.complex128), device=self.device)
        GS = torch.zeros_like(S)
        gmass = torch.zeros((self.m, self.n_angles, N), dtype=torch.float64, device=self.device)
        for j0 in self._ring_chunks(n_rows):
            X = S[j0:j0 + self._m3_ring_chunk].reshape(-1, S.shape[-1])
            om = ((dt[j0:j0 + self._m3_ring_chunk, None, None] * wt[None, None, :] / self.n_angles)
                  .expand(-1, self.n_angles, N).reshape(-1).to(torch.complex128))[:, None]
            G = torch.zeros_like(X)
            gm = torch.zeros(len(X), dtype=torch.float64, device=self.device)
            for block in self.basis.m3_blocks:
                a, b, c = block["q"]
                R = np.sqrt(block["weight"]) * r3[block["slice"]].reshape(block["shape"])
                x, y, z = X[:, self._stored[a]], X[:, self._stored[b]], X[:, self._stored[c]]
                Ta, Tb, Tc = block["shape"]
                # row-wise Khatri-Rao products times one matrix each: g_x = (conj y (x) z) R_(1)^T, g_y = (conj x (x) z) R_(2)^T,
                # g_z = (x (x) y) conj R_(3)
                kr = lambda u, v: (u[:, :, None] * v[:, None, :]).reshape(len(u), -1)
                gz = kr(x, y) @ torch.conj(R).reshape(Ta * Tb, Tc)
                G[:, self._stored[a]] += 2 * om * (kr(torch.conj(y), z) @ R.reshape(Ta, Tb * Tc).T)
                G[:, self._stored[b]] += 2 * om * (kr(torch.conj(x), z) @ R.permute(1, 0, 2).reshape(Tb, Ta * Tc).T)
                G[:, self._stored[c]] += 2 * om * gz
                gm += 2 * torch.real((gz * torch.conj(z)).sum(1))
            GS[j0:j0 + self._m3_ring_chunk] = G.reshape(-1, self.n_angles, N, S.shape[-1])
            gmass[j0:j0 + self._m3_ring_chunk] = gm.reshape(-1, self.n_angles, N)
        # back through s = K_j R_alpha x (complex linear) and x_{-p} = (-1)^p conj(c_p) (conjugate linear)
        GX = (torch.conj(self._phase)[:, None, :] * torch.einsum("jqn,javq->avn", self._K3, GS)).sum(0)      # (N, n)
        GC = torch.zeros((N, self.basis.n_coefficients), dtype=torch.complex128, device=self.device)
        start = 0
        for p, s in self._signed:
            block = GX[:, start:start + self.T[abs(p)]]
            GC[:, s] += block if p >= 0 else (-1.0) ** p * torch.conj(block)
            start += self.T[abs(p)]
        gw = torch.einsum("jai,j->i", gmass, dt) / self.n_angles
        return GC.cpu().numpy(), gw.cpu().numpy()

    def _m2_maps(self):
        """Zero-padded layout of the M2 operator: ``acc[j,q] = sum_p K[j,q,p] M_p K[j,q,p]^T`` for every q >= 0 and signed p in one einsum."""
        torch, Tm = self.torch, max(self.T)
        blocks = self.basis.m2_blocks
        n_stored = sum(int(b["slice"].stop - b["slice"].start) for b in blocks)
        src = np.full((len(self.orders), Tm, Tm), n_stored, dtype=np.int64)
        scale = np.zeros((len(self.orders), Tm, Tm))
        Kpad = np.zeros((self.m, self.L + 1, len(self.orders), Tm, Tm))
        K = self.K.cpu().numpy()
        for ip, p in enumerate(self.orders):
            pb, t = blocks[abs(p)], self.T[abs(p)]
            idx = np.arange(pb["slice"].start, pb["slice"].stop).reshape(t, t)
            src[ip, :t, :t] = idx if p >= 0 else idx.T
            scale[ip, :t, :t] = 1.0 / np.sqrt(pb["weight"])
            for q in range(self.L + 1):
                Kpad[:, q, ip, :self.T[q], :t] = K[:, self.off[q][0]:self.off[q][1], self.off[p][0]:self.off[p][1]]
        read, w = [], []
        for q, blk in enumerate(blocks):
            t = self.T[q]
            rows = (q * Tm + np.arange(t)[:, None]) * Tm + np.arange(t)[None, :]
            read.append(rows.ravel())
            w.append(np.full(t * t, np.sqrt(blk["weight"])))
        dev = self.device
        self._m2_src = torch.as_tensor(src, device=dev)
        self._m2_scale = torch.as_tensor(scale, dtype=torch.complex128, device=dev)
        self._Kpad = torch.as_tensor(Kpad, dtype=torch.complex128, device=dev)
        self._m2_read = torch.as_tensor(np.concatenate(read), dtype=torch.long, device=dev)
        self._m2_w = torch.as_tensor(np.concatenate(w), dtype=torch.float64, device=dev)

    # ---- operators --------------------------------------------------------- #
    def apply(self, order, m0):
        """``F[j]``: the ring-j moment of the flat stored moment ``m0`` (complex tensor on ``self.device``)."""
        torch = self.torch
        if order == 1:
            s0 = slice(*self.off[0])
            return torch.einsum("jab,b->ja", self.K[:, s0, s0].to(torch.complex128), m0)
        if order == 2:
            mp = torch.cat([m0, m0.new_zeros(1)])[self._m2_src] * self._m2_scale          # (orders, Tmax, Tmax), zero padded
            acc = torch.einsum("jqpab,pbc,jqpdc->jqad", self._Kpad, mp, self._Kpad)
            return acc.reshape(self.m, -1)[:, self._m2_read] * self._m2_w
        raise ValueError("M3 is evaluated from the images: use m3_rings and m3_gradients")

    def predict(self, order, m0, d):
        """``d @ F`` for numpy ``m0`` and ``d``; also returns ``F`` as numpy. The autograd graph is kept so that the following
        :meth:`pullback` of the same ``m0`` object does not repeat the forward pass."""
        torch = self.torch
        mr = torch.tensor(np.stack([m0.real, m0.imag], -1), dtype=torch.float64, device=self.device, requires_grad=True)
        F = self.apply(order, torch.view_as_complex(mr))
        self._graph = (mr, F)
        y = torch.tensor(np.asarray(d), dtype=torch.complex128, device=self.device) @ F.detach()
        return y.cpu().numpy(), F.detach().cpu().numpy()

    def pullback(self, d, clean):
        """Adjoint for the real inner product: gradient of ``sum <clean, d @ F(m0)>`` (real and imaginary parts) with respect to the ``m0``
        of the preceding :meth:`predict`, whose autograd graph is consumed here."""
        torch = self.torch
        mr, F = self._graph
        self._graph = None
        y = torch.tensor(np.asarray(d), dtype=torch.complex128, device=self.device) @ F
        c = torch.tensor(np.asarray(clean), dtype=torch.complex128, device=self.device)
        (g,) = torch.autograd.grad((torch.view_as_real(y) * torch.view_as_real(c)).sum(), mr)
        return (g[..., 0] + 1j * g[..., 1]).detach().cpu().numpy()


class _Law:
    """State shared by the wrapped moment functions and the joint fit."""

    def __init__(self, step_px, max_px, warm=None, finished=None, path=None):
        self.rho_px = ring_radii(step_px, max_px)
        self.warm = [np.asarray(w, float) for w in (warm or [])]
        self.records = list(finished or [])      # laws of the initializations a resumed stage has already finished
        self.calls = len(self.records)           # the initialization the next fit belongs to
        self.path = path
        self.order = 3
        self.rings = None
        self.active = False
        self.d = None
        self.F = None


@contextlib.contextmanager
def shift_law(step_px, max_px, warm=None, finished=None, path=None):
    """Fit the shift law jointly with volume and viewing density in the single-moment fit of the open context (any of M1, M2, M3).

    ``warm`` holds the fitted weights of the previous stage; initialization i starts from entry i. ``finished`` are the records of
    the initializations that were already fitted when the stage was interrupted. Yields the state object; its ``records`` list has one
    entry per initialization (ring radii in pixels, fitted weights, losses) and is written to ``path`` after every fit."""
    law = _Law(step_px, max_px, warm, finished, path)
    FB = _moments_module.FBMomentBasis
    orig_moments, orig_gradients = FB.moments, FB.moment_gradients
    orig_fit = _postprocessing.fit_single_moment_low_degree

    def moments(self, C, w, orders=(1, 2, 3), batch_size=4096):
        if law.active and law.order == 3 and 3 in orders:
            out = list(orig_moments(self, C, w, orders=tuple(o for o in orders if o != 3), batch_size=batch_size))
            law.F = law.rings.m3_rings(C, w)
            out[2] = np.asarray(law.d) @ law.F
            return tuple(out)
        out = list(orig_moments(self, C, w, orders=orders, batch_size=batch_size))
        k = law.order - 1
        if law.active and out[k] is not None:
            out[k], law.F = law.rings.predict(law.order, out[k], law.d)
        return tuple(out)

    def gradients(self, C, w, clean):
        if law.active and law.order == 3 and clean[2] is not None:
            G3, g3 = law.rings.m3_gradients(C, w, law.d, clean[2])
            G, g = orig_gradients(self, C, w, [clean[0], clean[1], None])
            return G + G3, g + g3
        k = law.order - 1
        if law.active and clean[k] is not None:
            clean = list(clean)
            clean[k] = law.rings.pullback(law.d, clean[k])
        return orig_gradients(self, C, w, clean)

    def fit(param, grid, targets, lambdas, theta0, beta0=None, degree=2, maxiter=5000, print_every=100):
        weights, law.order = _viewing._single_order_weights(lambdas)
        result = _joint_fit(law, param, grid, tuple(targets), weights, theta0, beta0, degree, int(maxiter), print_every)
        law.active = False
        return result

    FB.moments, FB.moment_gradients = moments, gradients
    _postprocessing.fit_single_moment_low_degree = fit
    try:
        yield law
    finally:
        FB.moments, FB.moment_gradients = orig_moments, orig_gradients
        _postprocessing.fit_single_moment_low_degree = orig_fit


def _joint_fit(law, param, grid, targets, weights, theta0, beta0, degree, maxiter, print_every):
    from scipy.optimize import minimize

    order = law.order
    w3 = np.broadcast_to(weights[order - 1], np.shape(targets[order - 1]))
    if law.rings is None:
        law.rings = ShiftRings(param.fb_basis, law.rho_px)
    M = len(law.rho_px)
    basis = param.basis      # symmetric volume: theta = basis @ phi (None without symmetry)
    nt = param.n_params if basis is None else basis.shape[1]
    expand = (lambda phi: phi) if basis is None else (lambda phi: basis @ phi)
    reduce = (lambda g: g) if basis is None else (lambda g: basis.T @ g)
    density = _viewing.LowDegreeViewingDensity(grid, degree=degree)
    b0 = density._beta(np.zeros(density.n_parameters) if beta0 is None else beta0)
    nb = len(b0)
    u0 = np.zeros(M)
    u0[0] = 1.0
    if law.calls < len(law.warm) and len(law.warm[law.calls]) == M:
        u0 = np.maximum(law.warm[law.calls], 0.0)
    law.calls += 1
    x0 = np.r_[reduce(np.asarray(theta0, float)), b0, u0]

    def objective(x):
        phi, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
        theta, total = expand(phi), u.sum()
        law.active, law.d = True, u / total
        value, g_theta, g_p, info = _moments_module.fb_objective_and_grad(
            theta, density.pmf(beta), param, *targets, lambdas=weights, return_info=True)
        r = np.asarray(info[f"masked_m{order}"]) - np.asarray(targets[order - 1])
        g_d = 2 * np.real(((w3 * np.conj(r))[None, :] * law.F.real.astype(complex)).sum(1))
        return value, np.r_[reduce(g_theta), density.vjp(beta, g_p), (g_d - law.d @ g_d) / total]

    state = dict(nit=0)

    def callback(_):
        state["nit"] += 1
        if state["nit"] % (print_every or 10 ** 9) == 0:
            print(f"    [m{order} shift law] it={state['nit']:5d}")

    result = minimize(objective, x0, jac=True, method="L-BFGS-B", callback=callback,
                      bounds=[(None, None)] * (nt + nb) + [(0.0, None)] * M,
                      options=dict(maxiter=maxiter, maxfun=max(15000, 20 * maxiter), ftol=1e-11, gtol=1e-10, maxcor=100))
    x = np.asarray(result.x, float).copy()
    phi, beta, u = x[:nt], x[nt:nt + nb], x[nt + nb:]
    theta, d, p = expand(phi), u / u.sum(), density.pmf(x[nt:nt + nb])
    # same final evaluation as fit_single_moment_low_degree
    law.active, law.d = True, d
    _, _, _, info = _moments_module.fb_objective_and_grad(theta, p, param, *targets, lambdas=weights, return_info=True)
    norm = max(float(np.linalg.norm(np.ravel(targets[order - 1]))), 1e-30)
    law.records.append(dict(rho_px=law.rho_px.tolist(), d=d.tolist(), loss=float(result.fun), nit=int(result.nit),
                            mean_radius_px=float((d * law.rho_px).sum()),
                            rms_radius_px=float(np.sqrt((d * law.rho_px ** 2).sum()))))
    print(f"  [m{order} shift law] loss={result.fun:.6e} nit={result.nit} mean radius {law.records[-1]['mean_radius_px']:.3f} px, "
        f"rms {law.records[-1]['rms_radius_px']:.3f} px")
    if law.path is not None:
        write_records(law, law.path)
    return dict(theta=theta.copy(), beta=beta.copy(), p=p.copy(), res=result,
                relative_error=float(info[f"misfit{order}"] / norm), viewing_diagnostics=density.evaluate(beta))


def write_records(law, path):
    """Save the fitted laws (one per initialization, in fit order)."""
    with open(path, "w") as f:
        json.dump(dict(rho_px=law.rho_px.tolist(),
                       initializations=law.records), f, indent=2)


def read_records(path):
    """The ``initializations`` list written by :func:`write_records`, or an empty list if the file does not exist."""
    path = Path(path)
    return json.loads(path.read_text())["initializations"] if path.is_file() else []


def select_m2_start(prepared_dir, m2_dir, output_dir, *, ell_max, viewing_degree, symmetry, device):
    """Pick the M2 initialization whose solution best explains M3, before any M3 fitting, and write it alone to ``output_dir``.

    For every M2 initialization the shifted model M3 of its volume, viewing density and shift weights is formed and given its
    least-squares amplitude (as the M3 warm start rescales the volume); the initialization with the smallest relative residual
    ||a Re M3_model - M3||^2 / ||M3||^2 is chosen. ``output_dir`` then holds ``init_0`` (that fit), its shift law and
    ``selection.json``, the layout of an M2 stage with one initialization, so M3 starts from it."""
    import shutil
    import time
    from .optimization import SphFBParam

    started = time.monotonic()
    m2_dir, output_dir = Path(m2_dir), Path(output_dir)
    _moments_module.use_device(device)
    prepared = _postprocessing.load_prepared_moments(prepared_dir, 3)
    grid = _moments_module.viewing_grid(3 * ell_max + viewing_degree)
    param = SphFBParam(prepared.side, ell_max, grid.rots, fb_basis=prepared.fb_basis, symmetry=symmetry)
    density = _viewing.LowDegreeViewingDensity(grid, degree=viewing_degree)
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
