"""The distribution of particle shifts.

The images are not centered individually. Their shifts follow an isotropic distribution on rings of radius ``rho_j`` (pixels of the moment
images) with weights ``d_j >= 0``, ``sum_j d_j = 1``:

    p(s) = sum_j d_j delta(|s| - rho_j) / (2 pi rho_j).

:class:`ShiftRings` gives the moments of the images shifted by each ring: M1 and M2 through analytic ring operators, M3 from the shifted
images themselves. :class:`ShiftLaw` holds the ring grid and the weights fitted in a stage; each stage starts from those of the previous one.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import jv

DEFAULT_SHIFT_STEP_PX = 0.5


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
        """``F[j]``: the ring-j moment of the flat stored M1 or M2 ``m0`` (complex tensor on ``self.device``)."""
        torch = self.torch
        if order == 1:
            s0 = slice(*self.off[0])
            return torch.einsum("jab,b->ja", self.K[:, s0, s0].to(torch.complex128), m0)
        mp = torch.cat([m0, m0.new_zeros(1)])[self._m2_src] * self._m2_scale          # order 2: (orders, Tmax, Tmax), zero padded
        acc = torch.einsum("jqpab,pbc,jqpdc->jqad", self._Kpad, mp, self._Kpad)
        return acc.reshape(self.m, -1)[:, self._m2_read] * self._m2_w

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


class ShiftLaw:
    """The ring grid and the fitted weights of one stage, one record per initialization in fit order, saved to ``path`` after every fit.

    ``warm`` holds the weights of the previous stage (initialization i starts from entry i); ``finished`` the records of the initializations that
    were fitted before the stage was interrupted (they are a prefix, and the next fit is numbered after them)."""

    def __init__(self, step_px, max_px, warm=(), finished=(), path=None):
        self.rho_px = ring_radii(step_px, max_px)
        self.warm = [np.asarray(w, float) for w in warm]
        self.records = list(finished)
        self.path = path
        self.rings = None

    def ring_operators(self, basis):
        """The ``ShiftRings`` of ``basis``, built on first use."""
        if self.rings is None:
            self.rings = ShiftRings(basis, self.rho_px)
        return self.rings

    def start(self):
        """Starting ring weights ``u`` of the next initialization: its warm start, otherwise all mass on ring 0 (no shift)."""
        index = len(self.records)
        if index < len(self.warm):
            return np.maximum(self.warm[index], 0.0)
        u0 = np.zeros(len(self.rho_px))
        u0[0] = 1.0
        return u0

    def record(self, d, loss, nit):
        """Append the fitted weights of the initialization just fitted and save all records."""
        self.records.append(dict(rho_px=self.rho_px.tolist(), d=d.tolist(), loss=float(loss), nit=int(nit),
                                 mean_radius_px=float((d * self.rho_px).sum()),
                                 rms_radius_px=float(np.sqrt((d * self.rho_px ** 2).sum()))))
        if self.path is not None:
            with open(self.path, "w") as f:
                json.dump(dict(rho_px=self.rho_px.tolist(), initializations=self.records), f, indent=2)
        return self.records[-1]


def read_records(path):
    """The ``initializations`` list saved by :meth:`ShiftLaw.record`, or an empty list if the file does not exist."""
    path = Path(path)
    return json.loads(path.read_text())["initializations"] if path.is_file() else []
