"""Check the shift-ring operators of kam.shift_distribution against explicit shifts.

Random Fourier-Bessel coefficients are made in-plane invariant by averaging over rotations, each image is shifted by a ring of fixed radius with
the angle averaged on a fine grid, and the resulting M1-M3 are compared with ShiftRings.apply; the adjoint is checked by finite differences.
usage: python tests/check_shift_distribution.py        (about a minute on one CPU node; prints ALL OK)"""
import numpy as np
import torch
from kam.moments import FBMomentBasis
from kam.shift_distribution import ShiftRings, ring_radii

rng = np.random.default_rng(0)
basis = FBMomentBasis(20, 7)
rho = np.array([0.0, 0.3, 1.1, 2.5])
rings = ShiftRings(basis, rho, device="cpu")
L = basis.ell_max
n_img, n_rot, n_psi = 10, 32, 32
C0 = rng.normal(size=(n_img, basis.n_coefficients)) + 1j * rng.normal(size=(n_img, basis.n_coefficients))
C0[:, basis.slices[0]] = C0[:, basis.slices[0]].real          # the order-0 coefficients of a real image are real


def rotate(C, g):
    out = C.copy()
    for q in range(L + 1):
        out[:, basis.slices[q]] *= np.exp(1j * q * g)
    return out


C = np.concatenate([rotate(C0, 2 * np.pi * i / n_rot) for i in range(n_rot)])
w = np.full(len(C), 1.0 / len(C))
m0 = basis.moments(C, w)


def shifted(C, j, psi):
    a = {}
    for q in range(L + 1):
        a[q] = C[:, basis.slices[q]]
        a[-q] = (-1) ** q * np.conj(a[q])
    K = rings.K[j].numpy()
    out = np.zeros_like(C)
    for q in range(L + 1):
        acc = 0
        for p in range(-L, L + 1):
            Kqp = K[rings.off[q][0]:rings.off[q][1], rings.off[p][0]:rings.off[p][1]]
            acc = acc + (-1j) ** (q - p) * np.exp(-1j * (q - p) * psi) * (a[p] @ Kqp.T)
        out[:, basis.slices[q]] = acc
    return out


ok = True
for order in (1, 2, 3):
    x = m0[order - 1]
    F = rings.apply(order, torch.tensor(x, dtype=torch.complex128)).numpy()
    for j in range(len(rho)):
        ref = sum(basis.moments(shifted(C, j, psi), w, orders=(order,))[order - 1] for psi in 2 * np.pi * np.arange(n_psi) / n_psi) / n_psi
        err = np.abs(F[j] - ref).max() / np.abs(ref).max()
        ok &= err < 1e-9
        print(f"M{order} ring rho={rho[j]:<4}: max relative error {err:.1e}")
d = np.array([0.1, 0.4, 0.3, 0.2])
for order in (1, 2, 3):
    x = m0[order - 1]
    cl = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    g = rings.pullback(order, x, d, cl)
    phi = lambda xx: float(np.real(np.vdot(cl, rings.predict(order, xx, d)[0])))
    dx = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    h = 1e-6
    fd, an = (phi(x + h * dx) - phi(x - h * dx)) / (2 * h), float(np.real(np.vdot(g, dx)))
    ok &= abs(fd - an) < 1e-5 * abs(fd)
    print(f"adjoint M{order}: finite difference {fd:.8e}  analytic {an:.8e}")
assert np.allclose(ring_radii(0.25, 5.0)[[0, 1, -1]], [0.0, 0.25, 5.0])
print("ALL OK" if ok else "FAILED")
raise SystemExit(0 if ok else 1)
