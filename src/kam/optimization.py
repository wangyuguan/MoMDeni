"""Real spherical Fourier--Bessel (SphFB) volume parameterization and its projection operator."""

import numpy as np
from . import utils
from scipy.special import spherical_jn
from scipy.optimize import brentq, minimize
from .moments import centered_coords_1d, centered_ifftn, _on_gpu


def _norm_assoc_legendre_all(nmax, x):
    """Evaluate normalized associated Legendre functions through degree ``nmax``."""
    x = np.asarray(x).ravel()
    y = np.zeros((nmax + 1, nmax + 1, x.size))
    u = -np.sqrt(np.maximum(0.0, (1.0 - x) * (1.0 + x)))
    y[0, 0] = 1.0
    for m in range(nmax + 1):
        if m:
            y[m, m] = y[m - 1, m - 1] * u * np.sqrt((2.0 * m - 1) / (2.0 * m))
        if m < nmax:
            y[m + 1, m] = x * y[m, m] * np.sqrt(2.0 * m + 1)
        for n in range(m + 2, nmax + 1):
            y[n, m] = ((2 * n - 1) * x * y[n - 1, m]
                       - np.sqrt((n + m - 1) * (n - m - 1)) * y[n - 2, m]) \
                      / np.sqrt((n - m) * (n + m))
    for n in range(nmax + 1):
        y[n, :n + 1] *= np.sqrt(2.0 * n + 1)
    return y


def _wigner_d(j, alpha, beta, gamma):
    """Return the degree-``j`` Wigner-D matrix in the adopted ZYZ convention."""
    fact = np.ones(2 * j + 1)
    for i in range(1, 2 * j + 1):
        fact[i] = fact[i - 1] * i
    D = np.zeros((2 * j + 1, 2 * j + 1), dtype=np.complex128)
    if abs(beta) < np.finfo(np.float32).eps:
        return np.diag(np.exp(-1j * (alpha + gamma) * np.arange(-j, j + 1)))
    for mp in range(-j, j + 1):
        for m in range(-j, j + 1):
            s = np.arange(max(0, m - mp), min(j + m, j - mp) + 1)
            scale = np.sqrt(fact[j + mp] * fact[j - mp] * fact[j + m] * fact[j - m])
            den = fact[j + m - s] * fact[mp - m + s] * fact[j - mp - s] * fact[s]
            d = ((-1.0) ** s * scale / den
                 * np.cos(beta / 2) ** (2 * j + m - mp - 2 * s)
                 * np.sin(beta / 2) ** (mp - m + 2 * s)).sum()
            D[mp + j, m + j] = np.exp(-1j * alpha * mp) * (-1) ** (mp - m) \
                                * d * np.exp(-1j * gamma * m)
    return D



def _small_d(j, betas):
    """Real factor (-1)^(mp-m) d^j_{mp m}(beta) of ``_wigner_d`` for an array of beta: shape (len(betas), 2j+1, 2j+1)."""
    fact = np.cumprod(np.r_[1.0, np.arange(1, 2 * j + 1)])
    c, s = np.cos(betas / 2), np.sin(betas / 2)
    out = np.zeros((len(betas), 2 * j + 1, 2 * j + 1))
    for mp in range(-j, j + 1):
        for m in range(-j, j + 1):
            scale = np.sqrt(fact[j + mp] * fact[j - mp] * fact[j + m] * fact[j - m])
            for k in range(max(0, m - mp), min(j + m, j - mp) + 1):
                den = fact[j + m - k] * fact[mp - m + k] * fact[j - mp - k] * fact[k]
                out[:, mp + j, m + j] += (-1.0) ** (mp - m + k) * scale / den * c ** (2 * j + m - mp - 2 * k) * s ** (mp - m + 2 * k)
    return out


def align_coefficients(target, moving, n_beta=90, n_angle=180):
    """Rotate (and possibly reflect) the complex SphFB coefficient blocks ``moving`` ({ell: (k, 2 ell + 1)}) onto ``target``.

    A rotation acts as A_ell -> A_ell conj(D_ell) (W_ell = conj(D_ell)^T, as in ``SphFBParam._rotated_coefficients``), so the inner product
    c(R) = Re sum_ell sum_{m,n} W_ell[m, n] C_ell[m, n] with C_ell = T_ell^H M_ell is a 2-D Fourier series in (alpha, gamma) for every beta:
    one FFT per beta on an n_beta x n_angle x n_angle grid, then Nelder-Mead from the grid maximum. A reflection is the inversion
    (A_ell times (-1)^ell) composed with a rotation. Returns the aligned blocks, the ZYZ angles, whether reflected and c."""
    betas = (np.arange(n_beta) + 0.5) * np.pi / n_beta
    dd = {ell: _small_d(ell, betas) for ell in target}
    best = None
    for reflected in (False, True):
        moved = {ell: (-1.0) ** ell * block for ell, block in moving.items()} if reflected else moving
        C = {ell: np.conj(target[ell]).T @ moved[ell] for ell in target}
        G = np.zeros((n_beta, n_angle, n_angle), dtype=np.complex128)        # G[beta, n mod n_angle, m mod n_angle]
        for ell in C:
            r = np.arange(-ell, ell + 1) % n_angle
            G[:, r[:, None], r[None, :]] += dd[ell] * C[ell].T[None]
        grid = np.real(np.fft.ifft2(G, axes=(1, 2)))
        ib, ia, ig = np.unravel_index(np.argmax(grid), grid.shape)

        def negative(x):
            return -sum(np.real(np.sum(np.conj(_wigner_d(ell, *x)).T * C[ell])) for ell in C)

        res = minimize(negative, [2 * np.pi * ia / n_angle, betas[ib], 2 * np.pi * ig / n_angle], method="Nelder-Mead",
                       options=dict(xatol=1e-6, fatol=1e-12, maxiter=2000))
        if best is None or -res.fun > best[3]:
            best = (moved, res.x, reflected, -res.fun)
    moved, angles, reflected, value = best
    return {ell: moved[ell] @ np.conj(_wigner_d(ell, *angles)) for ell in moved}, angles, reflected, value

def _rot_to_zyz(R):
    """Return angles for R = Rz(alpha) Ry(beta) Rz(gamma)."""
    if np.hypot(R[0, 2], R[1, 2]) > 1e-10:
        beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    elif R[2, 2] > 0:      # gimbal lock, beta = 0: R depends only on alpha + gamma
        beta, alpha, gamma = 0.0, np.arctan2(R[1, 0], R[0, 0]), 0.0
    else:                  # gimbal lock, beta = pi: R depends only on alpha - gamma
        beta, alpha, gamma = np.pi, np.arctan2(-R[1, 0], R[1, 1]), 0.0
    return alpha, beta, gamma


def _spherical_bessel_zeros(ell, r_max, step=np.pi / 20.0):
    """Zeros of the spherical Bessel function ``j_ell`` below ``r_max``.

    Sign changes are bracketed on a grid and refined with Brent's method. The
    default step is smaller than the minimum spacing between consecutive zeros.
    """
    if r_max <= 0:
        return np.zeros(0)
    xs = np.arange(1e-6, r_max + step, step)
    vals = spherical_jn(ell, xs)
    zeros = []
    for i in range(len(xs) - 1):
        fa, fb = vals[i], vals[i + 1]
        if fa == 0.0:
            zeros.append(xs[i])
        elif fa * fb < 0:
            zeros.append(brentq(lambda x: spherical_jn(ell, x), xs[i], xs[i + 1],
                                xtol=1e-13, rtol=8.9e-16))
    return np.array(sorted(z for z in zeros if z < r_max))


def _radial_truncation(ell_max, n):
    kmax, roots = [], []
    for ell in range(ell_max + 1):
        z = _spherical_bessel_zeros(ell, n * np.pi / 2)
        if len(z) == 0:
            break
        kmax.append(len(z))
        roots.append(z)
    return np.asarray(kmax, dtype=int), roots


def _real_to_complex(ell_max, kmax, indices):
    n = len(indices)
    S = np.zeros((n, n), dtype=np.complex128)
    for ell in range(ell_max + 1):
        for k in range(kmax[ell]):
            for m in range(-ell, ell + 1):
                i = indices[(ell, k, m)]
                if m > 0:
                    S[i, i] = 1
                    S[i, indices[(ell, k, -m)]] = -1j * (-1) ** (ell + m)
                elif m == 0:
                    S[i, i] = 1j ** ell
                else:
                    S[i, i] = 1j
                    S[i, indices[(ell, k, -m)]] = (-1) ** (ell + m)
    return S


class SphFBParam:
    """Real spherical Fourier--Bessel parameterization and projection operator.

    With ``fb_basis`` (:class:`moments.FBMomentBasis`), factors are precomputed for the fixed viewing rotations ``rots``
    and ``forward_images`` and ``adjoint_images`` are a real-linear adjoint pair between coefficient vectors and the
    steerable FB coefficients of the projections. Without it only the SphFB index layout is built.

    ``symmetry`` names a point group (see :mod:`kam.symmetry`). Then the volume coefficients are restricted to the symmetric subspace
    ``theta = basis @ phi`` (``basis`` is ``None`` for ``C1``, i.e. no restriction).
    """

    def __init__(self, L, ell_max, rots, fb_basis=None, symmetry="C1"):
        self.L = int(L)
        self.r_max = self.L / 2.0 - 1.0 if self.L % 2 == 0 else (self.L - 1) / 2
        self.ell_max = int(ell_max)
        self.rots = np.asarray(rots)
        self.fb_basis = fb_basis

        self.k_max, self.r0 = _radial_truncation(self.ell_max, self.L)
        self.ell_max = min(self.ell_max, len(self.k_max) - 1)
        self.indices = {}
        q = 0
        for ell in range(self.ell_max + 1):
            for k in range(self.k_max[ell]):
                for m in range(-ell, ell + 1):
                    self.indices[(ell, k, m)] = q
                    q += 1
        self.n_params = q
        self.S = _real_to_complex(self.ell_max, self.k_max, self.indices)
        self._blocks = {
            ell: np.stack([
                np.arange(self.indices[(ell, k, -ell)],
                          self.indices[(ell, k, ell)] + 1)
                for k in range(self.k_max[ell])
            ]) for ell in range(self.ell_max + 1)
        }
        self.symmetry = symmetry
        self.basis = None
        if symmetry != "C1":
            from .symmetry import invariant_basis
            self.basis = invariant_basis(self, symmetry)
        self._cart_basis = None
        self._gpu_factors = None
        if fb_basis is not None:
            self.B = fb_basis.slice_projection(self)
            self.W = self._rotation_blocks()
            if _on_gpu():
                self._upload_gpu_factors()

        c = centered_coords_1d(self.L)
        x, y, z = np.meshgrid(c, c, c, indexing="ij")
        self.ball_mask = np.sqrt(x*x + y*y + z*z) <= self.r_max

    def project(self, theta):
        """The coefficient vector of the symmetric subspace closest to ``theta`` (``theta`` itself for ``C1``)."""
        return theta if self.basis is None else self.basis @ (self.basis.T @ theta)

    def _basis(self, x, y, z):
        """Evaluate the complex basis at normalized Fourier coordinates."""
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        r = np.sqrt(x*x + y*y + z*z)
        theta = np.zeros_like(r)
        nz = r > 0
        theta[nz] = np.arccos(np.clip(z[nz] / r[nz], -1, 1))
        phi = np.arctan2(y, x)
        lp = _norm_assoc_legendre_all(self.ell_max, np.cos(theta)) / np.sqrt(4*np.pi)
        E = np.array([np.exp(1j * m * phi)
                      for m in range(-self.ell_max, self.ell_max + 1)])
        F = np.zeros((r.size, self.n_params), dtype=np.complex128)
        outside = r > 0.5
        for ell in range(self.ell_max + 1):
            Y = np.empty((r.size, 2 * ell + 1), dtype=np.complex128)
            for m in range(-ell, ell + 1):
                P = lp[ell, abs(m)]
                if m < 0:
                    P = (-1) ** m * P
                Y[:, m + ell] = P * E[m + self.ell_max]
            for k in range(self.k_max[ell]):
                root = self.r0[ell][k]
                radial = spherical_jn(ell, r * root / 0.5) \
                         * np.sqrt(2 / 0.5**3) / abs(spherical_jn(ell, root, True))
                radial[outside] = 0
                idx = self._blocks[ell][k]
                F[:, idx] = radial[:, None] * Y
        return F

    def _rotation_blocks(self):
        out = {}
        angles = [_rot_to_zyz(R) for R in self.rots]
        for ell in range(self.ell_max + 1):
            out[ell] = np.asarray([np.conj(_wigner_d(ell, *a)).T for a in angles])
        return out

    def _rotated_coefficients(self, theta):
        ac = self.S @ np.asarray(theta, dtype=self.S.real.dtype)
        u = np.zeros((len(self.rots), self.n_params), dtype=np.complex128)
        for ell, idx in self._blocks.items():
            A = ac[idx]
            u[:, idx] = np.einsum("imn,kn->ikm", self.W[ell], A, optimize=True)
        return u

    def _upload_gpu_factors(self):
        """Upload the factored projection operator to the active CUDA device."""
        torch = utils._gpu_torch()
        dev = utils._gpu_dev()
        self._gpu_factors = {
            "S": torch.as_tensor(np.ascontiguousarray(self.S), device=dev),
            "B": torch.as_tensor(np.ascontiguousarray(self.B), device=dev),
            "W": {ell: torch.as_tensor(np.ascontiguousarray(W), device=dev)
                  for ell, W in self.W.items()},
            "blocks": {ell: torch.as_tensor(idx, dtype=torch.long, device=dev)
                       for ell, idx in self._blocks.items()},
        }

    def _forward_gpu(self, theta):
        torch = utils._gpu_torch()
        dev = utils._gpu_dev()
        fac = self._gpu_factors
        th = torch.as_tensor(np.ascontiguousarray(theta), device=dev,
                             dtype=fac["S"].real.dtype)
        ac = fac["S"] @ th.to(fac["S"].dtype)
        u = torch.zeros((len(self.rots), self.n_params), dtype=fac["S"].dtype,
                        device=dev)
        for ell, idx in fac["blocks"].items():
            A = ac[idx]
            u[:, idx] = torch.einsum("imn,kn->ikm", fac["W"][ell], A)
        out = u @ fac["B"].T
        torch.cuda.synchronize()
        return out.cpu().numpy()

    def forward_images(self, theta):
        if self._gpu_factors is not None:
            return self._forward_gpu(theta)
        return self._rotated_coefficients(theta) @ self.B.T

    def adjoint_images(self, G):
        """Apply the real adjoint satisfying ``Re<G,F(dtheta)>=<out,dtheta>``."""
        if self._gpu_factors is not None:
            return self._adjoint_gpu(G)
        ubar = np.asarray(G, dtype=np.complex128) @ np.conj(self.B)
        acbar = np.zeros(self.n_params, dtype=np.complex128)
        for ell, idx in self._blocks.items():
            U = ubar[:, idx]
            Abar = np.einsum("imn,ikm->kn", np.conj(self.W[ell]), U,
                             optimize=True)
            np.add.at(acbar, idx, Abar)
        return np.real(np.conj(self.S).T @ acbar).astype(float)

    def _adjoint_gpu(self, G):
        torch = utils._gpu_torch()
        dev = utils._gpu_dev()
        fac = self._gpu_factors
        Gt = torch.as_tensor(np.ascontiguousarray(G), device=dev,
                             dtype=fac["B"].dtype)
        ubar = Gt @ torch.conj(fac["B"])
        acbar = torch.zeros(self.n_params, dtype=fac["S"].dtype, device=dev)
        for ell, idx in fac["blocks"].items():
            U = ubar[:, idx]
            Abar = torch.einsum("imn,ikm->kn", torch.conj(fac["W"][ell]), U)
            acbar[idx.reshape(-1)] += Abar.reshape(-1)
        out = torch.real(torch.conj(fac["S"]).T @ acbar)
        torch.cuda.synchronize()
        return out.cpu().numpy().astype(float, copy=False)

    def _cartesian_basis(self):
        """Complex SphFB basis at the Cartesian Fourier points of the ball (cached); ``embed`` applies ``S`` to the coefficients,
        which avoids the dense (points x params) @ (params x params) product (``S`` has at most two nonzeros per column)."""
        if self._cart_basis is None:
            c = centered_coords_1d(self.L) / self.L
            x, y, z = np.meshgrid(c, c, c, indexing="ij")
            mask = self.ball_mask.ravel()
            self._cart_basis = self._basis(x.ravel()[mask], y.ravel()[mask], z.ravel()[mask])
        return self._cart_basis

    def embed(self, theta):
        return self._embed_coefficients(self.S @ np.asarray(theta))

    def _embed_coefficients(self, ac):
        V = np.zeros(self.L ** 3, dtype=np.complex128)
        V[self.ball_mask.ravel()] = self._cartesian_basis() @ ac
        return V.reshape((self.L,) * 3)

    def volume(self, theta):
        """Return the real-space ``L^3`` volume represented by ``theta``."""
        return self.coefficient_volume(self.S @ np.asarray(theta))

    def coefficient_volume(self, ac):
        """Real-space ``L^3`` volume of the complex SphFB coefficients ``ac`` (``S @ theta``)."""
        return np.real(centered_ifftn(self._embed_coefficients(ac)))
