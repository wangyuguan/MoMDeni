"""Fourier--Bessel moment matching and SphFB volume reconstruction.

The objective combines Fourier--Bessel M1, M2, and M3 block residuals.
Block weights preserve complete-tensor Frobenius norms. Volume
coefficients and a viewing distribution are optimized against one active
moment order.
"""

import numpy as np
from . import utils
from scipy.special import spherical_jn
from scipy.optimize import brentq
from .moments import (
    centered_ifftn, DEFAULT_BATCH, centered_coords_1d,
    _iter_batches, _on_gpu,
)


def apply_fourier_image_shifts(images, offsets, ds_res, support=None,
                               batch_size=DEFAULT_BATCH):
    """Apply real-space translations to Fourier images in place.

    The phase is ``exp(-2*pi*i*k.dot(offset))``, where frequencies are in
    cycles per pixel and ``offsets`` has shape ``(N, 2)`` in downsampled-pixel
    units. ``images`` has shape ``(N, ds_res, ds_res)`` or ``(N, d)``. For the
    flattened form, ``support`` contains the retained Fortran-order pixel
    indices. The input array is returned after mutation.
    """
    images = np.asarray(images)
    offsets = np.asarray(offsets, dtype=float)
    if images.ndim not in (2, 3):
        raise ValueError("images must have shape (N,d) or (N,ds_res,ds_res)")
    if offsets.shape != (len(images), 2) or not np.all(np.isfinite(offsets)):
        raise ValueError("offsets must be a finite (N,2) array")
    if not np.issubdtype(images.dtype, np.complexfloating):
        raise TypeError("Fourier images must have a complex dtype")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    c = centered_coords_1d(ds_res) / float(ds_res)
    kx, ky = np.meshgrid(c, c, indexing="ij")
    freq = np.column_stack((kx.ravel(order="F"), ky.ravel(order="F")))
    if images.ndim == 3:
        if images.shape[1:] != (ds_res, ds_res) or support is not None:
            raise ValueError("full images must match ds_res and omit support")
    else:
        if support is not None:
            freq = freq[np.asarray(support, dtype=int)]
        if images.shape[1] != len(freq):
            raise ValueError("flattened image width does not match ds_res/support")
    for sl in _iter_batches(len(images), batch_size):
        phase = np.exp(-2j * np.pi * (offsets[sl] @ freq.T))
        if images.ndim == 3:
            phase = phase.reshape(-1, ds_res, ds_res, order="F")
        images[sl] *= phase
    return images


def _validate_contrast_scales(contrast_scales, n_orders):
    """Return one finite positive contrast multiplier per moment order."""
    if contrast_scales is None:
        return np.ones(n_orders, dtype=float)
    scales = np.asarray(contrast_scales, dtype=float)
    if (scales.shape != (n_orders,) or not np.all(np.isfinite(scales))
            or np.any(scales <= 0)):
        raise ValueError(
            f"contrast_scales must contain {n_orders} finite positive values")
    return scales


def _validate_lambdas(lambdas, targets):
    """Validate M1--M3 loss weights and structural weight arrays.

    Each ``lambdas[k]`` must be a scalar or shaped like ``targets[k]``.
    Arrays encode excluded entries and compressed-symmetry multiplicities.
    The validated weights are returned as a list because their shapes differ.
    """
    if len(lambdas) != len(targets):
        raise ValueError(
            f"lambdas must contain {len(targets)} entries, one per moment order")
    weights = []
    for k, (w, target) in enumerate(zip(lambdas, targets)):
        w = np.asarray(w, dtype=float)
        if w.shape not in ((), np.shape(target)):
            raise ValueError(f"m{k+1} weight must be scalar or shaped like m{k+1}")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(f"m{k+1} weight must be finite and nonnegative")
        weights.append(w)
    return weights


def joint_objective_and_grad(theta, p, param, grid, mbar1, mbar2, mbar3,
                             lambdas=None, return_info=False,
                             contrast_scales=None, reflection_augmented=False):
    """Evaluate the Fourier--Bessel M1--M3 loss and its gradients."""
    from .moments import fb_objective_and_grad

    if getattr(param, "fb_basis", None) is None:
        raise ValueError("moment matching requires a Fourier-Bessel basis")
    return fb_objective_and_grad(
        theta, p, param, grid, mbar1, mbar2, mbar3, lambdas=lambdas,
        return_info=return_info, contrast_scales=contrast_scales,
        reflection_augmented=reflection_augmented)


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


def _rot_to_zyz(R):
    """Return angles for R = Rz(alpha) Ry(beta) Rz(gamma)."""
    beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
    if abs(np.sin(beta)) > 1e-10:
        alpha = np.arctan2(R[1, 2], R[0, 2])
        gamma = np.arctan2(R[2, 1], -R[2, 0])
    else:
        alpha = np.arctan2(R[1, 0], R[0, 0])
        gamma = 0.0
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


def _image_support(L, support):
    """Validate integer indices into the Fortran-flattened ``L x L`` image."""
    if support is None:
        return np.arange(L * L)
    support = np.asarray(support)
    # A boolean pixel mask would otherwise be read as the indices 0 and 1.
    if not np.issubdtype(support.dtype, np.integer):
        raise ValueError("support must be an integer index array, not a mask")
    if (support.ndim != 1 or support.size == 0
            or support.min() < 0 or support.max() >= L * L):
        raise ValueError("support must index pixels of the L x L image grid")
    return support.astype(int)


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

    Factors are precomputed for a fixed ``SO(3)`` grid and image support.
    ``forward_images`` and ``adjoint_images`` are a real-linear adjoint pair and
    map between coefficient vectors and Fourier-image samples.
    """

    def __init__(self, L, ell_max, rots, support=None, dtype=np.complex128,
                 fb_basis=None):
        self.L = int(L)
        self.r_max = self.L / 2.0 - 1.0 if self.L % 2 == 0 else (self.L - 1) / 2
        self.ell_max = int(ell_max)
        self.rots = np.asarray(rots)
        # With ``fb_basis`` (:class:`moments.FBMomentBasis`), "images" are
        # steerable FB coefficients and ``support`` indexes those coefficients.
        self.fb_basis = fb_basis
        if fb_basis is None:
            self.support = _image_support(self.L, support)
        elif support is not None:
            raise ValueError("FB coefficient images do not take a pixel support")
        else:
            self.support = np.arange(fb_basis.n_coefficients)
        self.dtype = np.dtype(dtype)

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
        self.S = _real_to_complex(self.ell_max, self.k_max, self.indices).astype(dtype)
        self._blocks = {
            ell: np.stack([
                np.arange(self.indices[(ell, k, -ell)],
                          self.indices[(ell, k, ell)] + 1)
                for k in range(self.k_max[ell])
            ]) for ell in range(self.ell_max + 1)
        }
        if fb_basis is None:
            self.B = self._basis_on_slice().astype(dtype)
        else:
            self.B = fb_basis.slice_projection(self).astype(dtype)
        self.W = self._rotation_blocks(dtype)
        self._cart_basis = None
        self._gpu_factors = None
        if _on_gpu():
            self._upload_gpu_factors()

        c = centered_coords_1d(self.L)
        x, y, z = np.meshgrid(c, c, c, indexing="ij")
        self.ball_mask = np.sqrt(x*x + y*y + z*z) <= self.r_max

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

    def _basis_on_slice(self):
        c = centered_coords_1d(self.L) / self.L
        x, y = np.meshgrid(c, c, indexing="xy")
        xf, yf = x.ravel(order="F"), y.ravel(order="F")
        return self._basis(xf[self.support], yf[self.support],
                           np.zeros(len(self.support)))

    def _rotation_blocks(self, dtype):
        out = {}
        angles = [_rot_to_zyz(R) for R in self.rots]
        for ell in range(self.ell_max + 1):
            out[ell] = np.asarray([np.conj(_wigner_d(ell, *a)).T for a in angles],
                                  dtype=dtype)
        return out

    def _rotated_coefficients(self, theta):
        ac = self.S @ np.asarray(theta, dtype=self.S.real.dtype)
        u = np.zeros((len(self.rots), self.n_params), dtype=self.dtype)
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

    @property
    def projection_device(self):
        return "gpu" if self._gpu_factors is not None else "cpu"

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

    def forward_images(self, theta, rots=None, support=None):
        self._check_operator(rots, support)
        if self._gpu_factors is not None:
            return self._forward_gpu(theta)
        return self._rotated_coefficients(theta) @ self.B.T

    def adjoint_images(self, G, rots=None, support=None):
        """Apply the real adjoint satisfying ``Re<G,F(dtheta)>=<out,dtheta>``."""
        self._check_operator(rots, support)
        if self._gpu_factors is not None:
            return self._adjoint_gpu(G)
        ubar = np.asarray(G, dtype=self.dtype) @ np.conj(self.B)
        acbar = np.zeros(self.n_params, dtype=self.dtype)
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

    def _check_operator(self, rots, support):
        if rots is not None and (np.shape(rots) != np.shape(self.rots)
                                 or not np.allclose(rots, self.rots)):
            raise ValueError("SphFBParam is precomputed for a different rotation grid")
        if support is not None and not np.array_equal(support, self.support):
            raise ValueError("SphFBParam is precomputed for a different image support")

    def _cartesian_basis(self):
        if self._cart_basis is None:
            c = centered_coords_1d(self.L) / self.L
            x, y, z = np.meshgrid(c, c, c, indexing="ij")
            mask = self.ball_mask.ravel()
            self._cart_basis = self._basis(x.ravel()[mask], y.ravel()[mask],
                                           z.ravel()[mask]) @ self.S
        return self._cart_basis

    def embed(self, theta):
        V = np.zeros(self.L ** 3, dtype=np.complex128)
        V[self.ball_mask.ravel()] = self._cartesian_basis() @ np.asarray(theta)
        return V.reshape((self.L,) * 3)

    def reduce(self, Vhat):
        y = np.asarray(Vhat).ravel()[self.ball_mask.ravel()]
        A = self._cartesian_basis()
        # Hermitian targets constrain real coefficients through both components.
        return np.linalg.lstsq(np.vstack((A.real, A.imag)),
                               np.concatenate((y.real, y.imag)), rcond=1e-7)[0]

    def volume(self, theta):
        """Return the real-space ``L^3`` volume represented by ``theta``."""
        return np.real(centered_ifftn(self.embed(theta)))
