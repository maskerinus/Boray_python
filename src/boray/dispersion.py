"""Dispersion and Hamiltonian calculations for the BORAY port."""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from scipy.special import ive

from .constants import C2, EPSILON0, EV_TO_K, KB
from .models import EquilibriumPoint


def initial_kr_function(r: float, point: EquilibriumPoint, nphi: float, kz: float, w: float, qs: np.ndarray, ms: np.ndarray):
    """Create the scalar cold-plasma dispersion function used to solve the initial `k_r`."""
    w2 = w * w
    wcs = qs * point.B / ms
    wps2 = point.ns0 * qs**2 / (EPSILON0 * ms)

    eps1 = 1.0 - np.sum(wps2 / (w2 - wcs**2))
    eps2 = np.sum((wcs / w) * wps2 / (w2 - wcs**2))
    eps3 = 1.0 - np.sum(wps2 / w2)

    def kpar_sq_used(kr: float) -> float:
        return ((kr * point.Br + kz * point.Bz + nphi / r * point.Bphi) / point.B) ** 2

    def kper_sq_used(kr: float) -> float:
        return (kr**2 + kz**2 + nphi**2 / r**2) - kpar_sq_used(kr)

    def fDkr_used(kr: float) -> float:
        return (
            eps1 * (kper_sq_used(kr) * C2 / w2) ** 2
            - ((eps1 + eps3) * (eps1 - kpar_sq_used(kr) * C2 / w2) - eps2**2) * kper_sq_used(kr) * C2 / w2
            + eps3 * ((eps1 - kpar_sq_used(kr) * C2 / w2) ** 2 - eps2**2)
        )

    return fDkr_used


def solve_initial_kr(r: float, point: EquilibriumPoint, nphi: float, kz: float, w: float, qs: np.ndarray, ms: np.ndarray, guess: float) -> float:
    """Solve the initial radial wave number with SciPy's `fsolve`."""
    fDkr = initial_kr_function(r, point, nphi, kz, w, qs, ms)
    root = optimize.fsolve(lambda x: np.array([fDkr(float(x[0]))], dtype=float), x0=np.array([guess], dtype=float), xtol=1e-8)
    return float(root[0])


def colddr(qs: np.ndarray, ms: np.ndarray, fns0: np.ndarray, fB0: np.ndarray, fkx: np.ndarray, fkz: np.ndarray, fw: np.ndarray):
    """Python translation of `colddr.m`."""
    qs = np.asarray(qs, dtype=float).reshape(-1)
    ms = np.asarray(ms, dtype=float).reshape(-1)
    fns0 = np.asarray(fns0, dtype=float)
    fB0 = np.asarray(fB0, dtype=float).reshape(-1)
    fkx = np.asarray(fkx, dtype=float).reshape(-1)
    fkz = np.asarray(fkz, dtype=float).reshape(-1)
    fw = np.asarray(fw, dtype=complex).reshape(-1)

    npoint, S = fns0.shape
    ww = np.zeros(npoint, dtype=complex)
    detD = np.zeros(npoint, dtype=complex)

    for jp in range(npoint):
        ns0 = fns0[jp]
        B0 = fB0[jp]
        kx = fkx[jp]
        kz = fkz[jp]
        w = fw[jp]

        wps = np.sqrt(ns0 * qs**2 / ms / EPSILON0)
        wcs = B0 * qs / ms
        wps2 = wps**2

        NN = 3 * S + 6
        SJ = 3 * S
        M = lil_matrix((NN, NN), dtype=complex)
        for s in range(S):
            ind = 3 * s
            M[ind + 1, ind + 0] += -1j * wcs[s]
            M[ind + 0, ind + 1] += 1j * wcs[s]
            M[ind + 0, SJ + 0] += 1j * qs[s] / ms[s]
            M[ind + 1, SJ + 1] += 1j * qs[s] / ms[s]
            M[ind + 2, SJ + 2] += 1j * qs[s] / ms[s]
            M[SJ + 0, ind + 0] += -1j * qs[s] * ns0[s] / EPSILON0
            M[SJ + 1, ind + 1] += -1j * qs[s] * ns0[s] / EPSILON0
            M[SJ + 2, ind + 2] += -1j * qs[s] * ns0[s] / EPSILON0

        M[SJ + 0, SJ + 4] += C2 * kz
        M[SJ + 1, SJ + 3] += -C2 * kz
        M[SJ + 1, SJ + 5] += C2 * kx
        M[SJ + 2, SJ + 4] += -C2 * kx
        M[SJ + 3, SJ + 1] += -kz
        M[SJ + 4, SJ + 0] += kz
        M[SJ + 4, SJ + 2] += -kx
        M[SJ + 5, SJ + 1] += kx

        ww[jp] = eigs(M.tocsr(), k=1, sigma=w, return_eigenvectors=False)[0]

        w2 = w * w
        eps1 = 1.0 - np.sum(wps2 / (w2 - wcs**2))
        eps2 = np.sum((wcs / w) * wps2 / (w2 - wcs**2))
        eps3 = 1.0 - np.sum(wps2 / w2)
        kpar2 = kz * kz
        kper2 = kx * kx
        detD[jp] = (
            eps1 * (kper2 * C2 / w2) ** 2
            - ((eps1 + eps3) * (eps1 - kpar2 * C2 / w2) - eps2**2) * kper2 * C2 / w2
            + eps3 * ((eps1 - kpar2 * C2 / w2) ** 2 - eps2**2)
        )

    return ww, detD


def dydt(
    r: float,
    phi: float,
    z: float,
    kr: float,
    nphi: float,
    kz: float,
    point: EquilibriumPoint,
    qs: np.ndarray,
    ms: np.ndarray,
    w: float,
    c2: float = C2,
):
    """Python translation of the active `jnew=2` branch in `dydt.m`."""
    del phi, z
    w2 = w * w

    B = point.B
    Br = point.Br
    Bz = point.Bz
    Bphi = point.Bphi

    kpar = (kr * Br + kz * Bz + nphi / r * Bphi) / B
    kpar2 = kpar * kpar
    k2 = kr * kr + kz * kz + nphi * nphi / (r * r)
    kper2 = k2 - kpar2

    dkpardr = -kpar / B * point.dBdr + (kr * point.dBrdr + kz * point.dBzdr + nphi / r * point.dBphidr - Bphi * nphi / r**2) / B
    dkpardz = -kpar / B * point.dBdz + (kr * point.dBrdz + kz * point.dBzdz + nphi / r * point.dBphidz) / B

    wcs = qs * B / ms
    wps2 = point.ns0 * qs**2 / (EPSILON0 * ms)
    dwps2dr = point.dns0dr * qs**2 / (EPSILON0 * ms)
    dwps2dz = point.dns0dz * qs**2 / (EPSILON0 * ms)
    dwcsdr = qs / ms * point.dBdr
    dwcsdz = qs / ms * point.dBdz

    ind = point.ns0 != 0.0
    wcs = wcs[ind]
    wps2 = wps2[ind]
    dwps2dr = dwps2dr[ind]
    dwps2dz = dwps2dz[ind]
    dwcsdr = dwcsdr[ind]
    dwcsdz = dwcsdz[ind]

    if wcs.size == 0:
        Ys = np.array([1.0])
        ja = np.array([0])
        ja1 = 0
        jb = np.array([], dtype=int)
        Ya = 1.0
        eps1 = 1.0
        eps2 = 0.0
        eps3 = 1.0
        G1 = 1.0
        G2 = (eps1 + eps3) * (eps1 - kpar2 * c2 / w2)
        G3 = eps3 * (eps1 - kpar2 * c2 / w2) ** 2
        dYadr = 0.0
        dYadz = 0.0
        deps1dr = 0.0
        deps1dz = 0.0
        deps2dr = 0.0
        deps2dz = 0.0
        deps3dr = 0.0
        deps3dz = 0.0
        dG1dr = 0.0
        dG1dz = 0.0
        dG2dr = 0.0
        dG2dz = 0.0
        dG3dr = 0.0
        dG3dz = 0.0
    else:
        Ys = np.abs(1.0 - wcs**2 / w2)
        ja = np.where(Ys == np.min(Ys))[0]
        ja1 = int(ja[0])
        jb = np.where(Ys > np.min(Ys))[0]
        Ya = 1.0 - wcs[ja1] ** 2 / w2

        eps1 = 1.0 - np.sum(wps2[jb] / (w2 - wcs[jb] ** 2))
        eps2 = np.sum((wcs[jb] / w) * wps2[jb] / (w2 - wcs[jb] ** 2))
        eps3 = 1.0 - np.sum(wps2 / w2)
        G1 = Ya * eps1 - np.sum(wps2[ja] / w2)
        G2 = (
            Ya * (eps1 + eps3) * (eps1 - kpar2 * c2 / w2)
            - np.sum(wps2[ja] / w2) * (2.0 * eps1 + eps3 - kpar2 * c2 / w2)
            - Ya * eps2**2
            - 2.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + np.sum(wps2[ja] / w2) ** 2
        )
        G3 = eps3 * (
            Ya * (eps1 - kpar2 * c2 / w2) ** 2
            - 2.0 * np.sum(wps2[ja] / w2) * (eps1 - kpar2 * c2 / w2)
            - Ya * eps2**2
            - 2.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + np.sum(wps2[ja] / w2) ** 2
        )

        dYadr = -2.0 * wcs[ja1] * dwcsdr[ja1] / w2
        dYadz = -2.0 * wcs[ja1] * dwcsdz[ja1] / w2
        deps1dr = -np.sum(dwps2dr[jb] / (w2 - wcs[jb] ** 2)) - 2.0 * np.sum(
            wps2[jb] / (w2 - wcs[jb] ** 2) ** 2 * wcs[jb] * dwcsdr[jb]
        )
        deps1dz = -np.sum(dwps2dz[jb] / (w2 - wcs[jb] ** 2)) - 2.0 * np.sum(
            wps2[jb] / (w2 - wcs[jb] ** 2) ** 2 * wcs[jb] * dwcsdz[jb]
        )
        deps2dr = (
            np.sum((dwcsdr[jb] / w) * wps2[jb] / (w2 - wcs[jb] ** 2))
            + np.sum((wcs[jb] / w) * dwps2dr[jb] / (w2 - wcs[jb] ** 2))
            + 2.0 * np.sum((wcs[jb] / w) * wps2[jb] / (w2 - wcs[jb] ** 2) ** 2 * wcs[jb] * dwcsdr[jb])
        )
        deps2dz = (
            np.sum((dwcsdz[jb] / w) * wps2[jb] / (w2 - wcs[jb] ** 2))
            + np.sum((wcs[jb] / w) * dwps2dz[jb] / (w2 - wcs[jb] ** 2))
            + 2.0 * np.sum((wcs[jb] / w) * wps2[jb] / (w2 - wcs[jb] ** 2) ** 2 * wcs[jb] * dwcsdz[jb])
        )
        deps3dr = -np.sum(dwps2dr / w2)
        deps3dz = -np.sum(dwps2dz / w2)

        dG1dr = dYadr * eps1 + Ya * deps1dr - np.sum(dwps2dr[ja] / w2)
        dG1dz = dYadz * eps1 + Ya * deps1dz - np.sum(dwps2dz[ja] / w2)
        dG2dr = (
            (dYadr * (eps1 + eps3) + Ya * (deps1dr + deps3dr)) * (eps1 - kpar2 * c2 / w2)
            + Ya * (eps1 + eps3) * deps1dr
            - np.sum(dwps2dr[ja] / w2) * (2.0 * eps1 + eps3 - kpar2 * c2 / w2)
            - np.sum(wps2[ja] / w2) * (2.0 * deps1dr + deps3dr)
            - dYadr * eps2**2
            - 2.0 * Ya * deps2dr * eps2
            - 2.0 * deps2dr * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(dwcsdr[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(wcs[ja] * dwps2dr[ja] / w2 / w)
            + 2.0 * np.sum(dwps2dr[ja] / w2) * np.sum(wps2[ja] / w2)
        )
        dG2dz = (
            (dYadz * (eps1 + eps3) + Ya * (deps1dz + deps3dz)) * (eps1 - kpar2 * c2 / w2)
            + Ya * (eps1 + eps3) * deps1dz
            - np.sum(dwps2dz[ja] / w2) * (2.0 * eps1 + eps3 - kpar2 * c2 / w2)
            - np.sum(wps2[ja] / w2) * (2.0 * deps1dz + deps3dz)
            - dYadz * eps2**2
            - 2.0 * Ya * deps2dz * eps2
            - 2.0 * deps2dz * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(dwcsdz[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(wcs[ja] * dwps2dz[ja] / w2 / w)
            + 2.0 * np.sum(dwps2dz[ja] / w2) * np.sum(wps2[ja] / w2)
        )
        dG3dr = deps3dr * (
            Ya * (eps1 - kpar2 * c2 / w2) ** 2
            - 2.0 * np.sum(wps2[ja] / w2) * (eps1 - kpar2 * c2 / w2)
            - Ya * eps2**2
            - 2.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + np.sum(wps2[ja] / w2) ** 2
        ) + eps3 * (
            dYadr * (eps1 - kpar2 * c2 / w2) ** 2
            + 2.0 * (Ya * deps1dr - np.sum(dwps2dr[ja] / w2)) * (eps1 - kpar2 * c2 / w2)
            - 2.0 * np.sum(wps2[ja] / w2) * deps1dr
            - dYadr * eps2**2
            - 2.0 * Ya * deps2dr * eps2
            - 2.0 * deps2dr * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(dwcsdr[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(wcs[ja] * dwps2dr[ja] / w2 / w)
            + 2.0 * np.sum(dwps2dr[ja] / w2) * np.sum(wps2[ja] / w2)
        )
        dG3dz = deps3dz * (
            Ya * (eps1 - kpar2 * c2 / w2) ** 2
            - 2.0 * np.sum(wps2[ja] / w2) * (eps1 - kpar2 * c2 / w2)
            - Ya * eps2**2
            - 2.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + np.sum(wps2[ja] / w2) ** 2
        ) + eps3 * (
            dYadz * (eps1 - kpar2 * c2 / w2) ** 2
            + 2.0 * (Ya * deps1dz - np.sum(dwps2dz[ja] / w2)) * (eps1 - kpar2 * c2 / w2)
            - 2.0 * np.sum(wps2[ja] / w2) * deps1dz
            - dYadz * eps2**2
            - 2.0 * Ya * deps2dz * eps2
            - 2.0 * deps2dz * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(dwcsdz[ja] * wps2[ja] / w2 / w)
            - 2.0 * eps2 * np.sum(wcs[ja] * dwps2dz[ja] / w2 / w)
            + 2.0 * np.sum(dwps2dz[ja] / w2) * np.sum(wps2[ja] / w2)
        )

    dDdkpar2 = kper2 * (c2 / w2) ** 2 * (Ya * eps1 - np.sum(wps2[ja] / w2) + Ya * eps3) - 2.0 * eps3 * (
        Ya * eps1 - np.sum(wps2[ja] / w2) - Ya * kpar2 * c2 / w2
    ) * c2 / w2
    dDdkper2 = 2.0 * G1 * kper2 * (c2 / w2) ** 2 - G2 * c2 / w2
    dDdr = dG1dr * (kper2 * c2 / w2) ** 2 - dG2dr * kper2 * c2 / w2 + dG3dr
    dDdz = dG1dz * (kper2 * c2 / w2) ** 2 - dG2dz * kper2 * c2 / w2 + dG3dz
    D = G1 * (kper2 * c2 / w2) ** 2 - G2 * kper2 * c2 / w2 + G3

    dFdkr = 2.0 * (dDdkpar2 - dDdkper2) * kpar * Br / B + 2.0 * dDdkper2 * kr
    dFdnphi = 2.0 * (dDdkpar2 - dDdkper2) * kpar * Bphi / (r * B) + 2.0 * dDdkper2 * nphi / r**2
    dFdkz = 2.0 * (dDdkpar2 - dDdkper2) * kpar * Bz / B + 2.0 * dDdkper2 * kz
    dFdr = dDdr + 2.0 * (dDdkpar2 - dDdkper2) * kpar * dkpardr - 2.0 * dDdkper2 * nphi**2 / r**3
    dFdphi = 0.0
    dFdz = dDdz + 2.0 * (dDdkpar2 - dDdkper2) * kpar * dkpardz

    if wcs.size == 0:
        deps1dw = 0.0
        deps2dw = 0.0
        deps3dw = 0.0
        dYadw = 0.0
        dG1dw = 0.0
        dG2dw = 0.0
        dG3dw = 0.0
    else:
        deps1dw = 2.0 * w * np.sum(wps2[jb] / (w2 - wcs[jb] ** 2) ** 2)
        deps2dw = -(
            np.sum(wcs[jb] / w**2 * wps2[jb] / (w2 - wcs[jb] ** 2))
            + 2.0 * np.sum(wps2[jb] / (w2 - wcs[jb] ** 2) ** 2 * wcs[jb])
        )
        deps3dw = 2.0 * np.sum(wps2 / w**3)
        dYadw = 2.0 * wcs[ja1] ** 2 / w**3

        dG1dw = dYadw * eps1 + Ya * deps1dw + 2.0 * np.sum(wps2[ja] / w2) / w
        dG2dw = (
            dYadw * (eps1 + eps3) * (eps1 - kpar2 * c2 / w2)
            + Ya * (deps1dw + deps3dw) * (eps1 - kpar2 * c2 / w2)
            + Ya * (eps1 + eps3) * (deps1dw + 2.0 * kpar2 * c2 / w2 / w)
            - np.sum(wps2[ja] / w2) * (2.0 * deps1dw + deps3dw + 2.0 * kpar2 * c2 / w2 / w)
            + 2.0 * np.sum(wps2[ja] / w2) / w * (2.0 * eps1 + eps3 - kpar2 * c2 / w2)
            - dYadw * eps2**2
            - 2.0 * Ya * eps2 * deps2dw
            - 2.0 * deps2dw * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + 6.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w) / w
            - 4.0 * np.sum(wps2[ja] / w2) * np.sum(wps2[ja] / w2) / w
        )
        dG3dw = deps3dw * (
            Ya * (eps1 - kpar2 * c2 / w2) ** 2
            - 2.0 * np.sum(wps2[ja] / w2) * (eps1 - kpar2 * c2 / w2)
            - Ya * eps2**2
            - 2.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + np.sum(wps2[ja] / w2) ** 2
        ) + eps3 * (
            dYadw * (eps1 - kpar2 * c2 / w2) ** 2
            + 2.0 * (Ya * (eps1 - kpar2 * c2 / w2) - np.sum(wps2[ja] / w2)) * (deps1dw + 2.0 * kpar2 * c2 / w2 / w)
            + 4.0 * np.sum(wps2[ja] / w2) / w * (eps1 - kpar2 * c2 / w2)
            - dYadw * eps2**2
            - 2.0 * Ya * eps2 * deps2dw
            - 2.0 * deps2dw * np.sum(wcs[ja] * wps2[ja] / w2 / w)
            + 6.0 * eps2 * np.sum(wcs[ja] * wps2[ja] / w2 / w) / w
            - 4.0 * np.sum(wps2[ja] / w2) * np.sum(wps2[ja] / w2) / w
        )

    dFdw = dG1dw * (kper2 * c2 / w2) ** 2 - 4.0 * G1 * (kper2 * c2 / w2) ** 2 / w - dG2dw * kper2 * c2 / w2 + 2.0 * G2 * kper2 * c2 / w2 / w + dG3dw
    return kpar, kper2, dFdr, dFdphi, dFdz, dFdkr, dFdnphi, dFdkz, dFdw, D


def hotdreach(
    qs: np.ndarray,
    ms: np.ndarray,
    fns0: np.ndarray,
    fTs0: np.ndarray,
    fB0: np.ndarray,
    fkx: np.ndarray,
    fkz: np.ndarray,
    fw: np.ndarray,
    jeach: int,
    joutw: int,
    N: int,
    J: int,
):
    """Python translation of `hotdreach.m`."""
    qs = np.asarray(qs, dtype=float).reshape(-1)
    ms = np.asarray(ms, dtype=float).reshape(-1)
    fns0 = np.asarray(fns0, dtype=float)
    fTs0 = np.asarray(fTs0, dtype=float)
    fB0 = np.asarray(fB0, dtype=float).reshape(-1)
    fkx = np.asarray(fkx, dtype=float).reshape(-1)
    fkz = np.asarray(fkz, dtype=float).reshape(-1)
    fw = np.asarray(fw, dtype=complex).reshape(-1)

    bzj, czj = _j_pole_coefficients(J)
    npoint, S = fns0.shape
    if jeach == 0:
        ww_each = np.zeros((npoint, 1), dtype=complex)
        detD_each = np.zeros((npoint, 1), dtype=complex)
        N_each = 1
    else:
        ww_each = np.zeros((npoint, 1 + S), dtype=complex)
        detD_each = np.zeros((npoint, 1 + S), dtype=complex)
        N_each = 1 + S

    for jp in range(npoint):
        ns0 = fns0[jp]
        Ts0 = fTs0[jp]
        B0 = fB0[jp]
        kx = fkx[jp]
        kz = fkz[jp]
        w0 = fw[jp]

        for js in range(N_each):
            w = ww_each[jp - 1, js] if jp > 0 else w0
            if js == 0:
                Ts = Ts0 * EV_TO_K
            else:
                Ts = np.full_like(Ts0, EV_TO_K)
                Ts[js - 1] = Ts0[js - 1] * EV_TO_K

            vtzs = np.sqrt(2.0 * KB * Ts / ms)
            wps = np.sqrt(ns0 * qs**2 / ms / EPSILON0)
            wcs = B0 * qs / ms
            rhocs = np.sqrt(KB * Ts / ms) / wcs
            wps2 = wps**2

            SNJ = S * (2 * N + 1) * J
            SNJ1 = SNJ
            SNJ3 = 3 * SNJ1
            NN = SNJ3 + 6

            bs = kx * rhocs
            bs[np.abs(bs) < 1e-50] = 1e-50
            bs2 = bs**2

            csnj = np.zeros(SNJ, dtype=complex)
            b11snj = np.zeros(SNJ, dtype=complex)
            b12snj = np.zeros(SNJ, dtype=complex)
            b13snj = np.zeros(SNJ, dtype=complex)
            b21snj = np.zeros(SNJ, dtype=complex)
            b22snj = np.zeros(SNJ, dtype=complex)
            b23snj = np.zeros(SNJ, dtype=complex)
            b31snj = np.zeros(SNJ, dtype=complex)
            b32snj = np.zeros(SNJ, dtype=complex)
            b33snj = np.zeros(SNJ, dtype=complex)

            snj = 0
            for s in range(S):
                for n in range(-N, N + 1):
                    Gamn = ive(n, bs2[s])
                    Gamnp = (ive(n + 1, bs2[s]) + ive(n - 1, bs2[s]) - 2.0 * ive(n, bs2[s])) / 2.0
                    for j in range(J):
                        csnj[snj] = czj[j] * kz * vtzs[s] + n * wcs[s]
                        tmp = wps2[s] * bzj[j]
                        b11snj[snj] = tmp * n**2 * Gamn / bs2[s]
                        b12snj[snj] = tmp * 1j * n * Gamnp
                        b21snj[snj] = -b12snj[snj]
                        b22snj[snj] = tmp * (n**2 * Gamn / bs2[s] - 2.0 * bs2[s] * Gamnp)
                        b13snj[snj] = tmp * czj[j] * n * math.sqrt(2.0) * Gamn / bs[s]
                        b31snj[snj] = b13snj[snj]
                        b23snj[snj] = -1j * tmp * czj[j] * math.sqrt(2.0) * Gamnp * bs[s]
                        b32snj[snj] = -b23snj[snj]
                        b33snj[snj] = tmp * (czj[j] ** 2) * 2.0 * Gamn
                        snj += 1

            if joutw == 1:
                M = lil_matrix((NN, NN), dtype=complex)
                for snj in range(SNJ):
                    jjx = snj
                    jjy = snj + SNJ1
                    jjz = snj + 2 * SNJ1
                    M[jjx, jjx] += csnj[snj]
                    M[jjx, SNJ3 + 0] += b11snj[snj]
                    M[jjx, SNJ3 + 1] += b12snj[snj]
                    M[jjx, SNJ3 + 2] += b13snj[snj]

                    M[jjy, jjy] += csnj[snj]
                    M[jjy, SNJ3 + 0] += b21snj[snj]
                    M[jjy, SNJ3 + 1] += b22snj[snj]
                    M[jjy, SNJ3 + 2] += b23snj[snj]

                    M[jjz, jjz] += csnj[snj]
                    M[jjz, SNJ3 + 0] += b31snj[snj]
                    M[jjz, SNJ3 + 1] += b32snj[snj]
                    M[jjz, SNJ3 + 2] += b33snj[snj]

                M[SNJ3 + 0, 0:SNJ1] = -1.0
                M[SNJ3 + 1, SNJ1 : 2 * SNJ1] = -1.0
                M[SNJ3 + 2, 2 * SNJ1 : 3 * SNJ1] = -1.0
                M[SNJ3 + 0, SNJ3 + 4] += C2 * kz
                M[SNJ3 + 1, SNJ3 + 3] += -C2 * kz
                M[SNJ3 + 1, SNJ3 + 5] += C2 * kx
                M[SNJ3 + 2, SNJ3 + 4] += -C2 * kx
                M[SNJ3 + 3, SNJ3 + 1] += -kz
                M[SNJ3 + 4, SNJ3 + 0] += kz
                M[SNJ3 + 4, SNJ3 + 2] += -kx
                M[SNJ3 + 5, SNJ3 + 1] += kx
                ww_each[jp, js] = eigs(M.tocsr(), k=1, sigma=w, return_eigenvectors=False)[0]
            else:
                ww_each[jp, js] = w

            wd = w
            denom = wd - csnj
            sigmawk = np.zeros((3, 3), dtype=complex)
            sigmawk[0, 0] = np.sum(b11snj / denom)
            sigmawk[0, 1] = np.sum(b12snj / denom)
            sigmawk[0, 2] = np.sum(b13snj / denom)
            sigmawk[1, 0] = np.sum(b21snj / denom)
            sigmawk[1, 1] = np.sum(b22snj / denom)
            sigmawk[1, 2] = np.sum(b23snj / denom)
            sigmawk[2, 0] = np.sum(b31snj / denom)
            sigmawk[2, 1] = np.sum(b32snj / denom)
            sigmawk[2, 2] = np.sum(b33snj / denom)
            sigmawk = -1j * EPSILON0 * sigmawk

            Qwk = -sigmawk / (1j * wd * EPSILON0)
            Kwk = np.eye(3, dtype=complex) + Qwk
            k_outer = np.array([[kx * kx, 0.0, kx * kz], [0.0, 0.0, 0.0], [kz * kx, 0.0, kz * kz]], dtype=complex)
            Dwk = Kwk + (k_outer / (kx**2 + kz**2) - np.eye(3, dtype=complex)) * ((kx**2 + kz**2) * C2 / wd**2)
            detD_each[jp, js] = np.linalg.det(Dwk)

    return ww_each, detD_each, N_each


def _j_pole_coefficients(J: int):
    if J == 8:
        bzj = np.array(
            [
                -0.017340112270401 - 0.046306439626294j,
                -0.739917811220052 + 0.839518284620274j,
                5.840632105105495 + 0.953602751322040j,
                -5.583374181615043 - 11.208550459628098j,
            ],
            dtype=complex,
        )
        czj = np.array(
            [
                2.237687725134293 - 1.625941024120362j,
                1.465234091939142 - 1.789620299603315j,
                0.839253966367922 - 1.891995211531426j,
                0.273936218055381 - 1.941787037576095j,
            ],
            dtype=complex,
        )
        return np.concatenate([bzj, np.conj(bzj)]), np.concatenate([czj, -np.conj(czj)])
    if J == 12:
        bzj = np.array(
            [
                -10.020983259474214 - 14.728932929429875j,
                -0.5887816915344951 + 0.19067303610080007j,
                -0.27475707659732384 + 3.6179207174938845j,
                0.00045713742777499515 + 0.00027155393843737099j,
                0.01794062703250838 - 0.03643605327670125j,
                10.36612426314575 - 2.5069048649816146j,
            ],
            dtype=complex,
        )
        czj = np.array(
            [
                0.22660012611958089 - 2.071687759489779j,
                -1.700292151630035 - 1.8822474221612724j,
                1.1713932508560118 - 1.977250331920854j,
                3.066620112682697 - 1.5900208259325997j,
                2.3073274904105783 - 1.75467325437282j,
                0.6872005249060191 - 2.040288525975844j,
            ],
            dtype=complex,
        )
        return np.concatenate([bzj, np.conj(bzj)]), np.concatenate([czj, -np.conj(czj)])
    if J == 4:
        bzj = np.array([0.546796859834032 + 0.037196505239277j, -1.046796859834027 + 2.101852568038518j], dtype=complex)
        czj = np.array([1.23588765343592 - 1.21498213255731j, -0.378611612386277 - 1.35094358543273j], dtype=complex)
        return np.concatenate([bzj, np.conj(bzj)]), np.concatenate([czj, -np.conj(czj)])
    if J == 3:
        bzj = np.array([0.1822 + 0.5756j, -1.3643, 0.1822 - 0.5756j], dtype=complex)
        czj = np.array([-0.9217 - 0.9091j, -1.0204j, 0.9217 - 0.9091j], dtype=complex)
        return bzj, czj
    if J == 2:
        bzj = np.array([-(0.5 + 0.81j), -(0.5 - 0.81j)], dtype=complex)
        czj = np.array([0.51 - 0.81j, -0.51 - 0.81j], dtype=complex)
        return bzj, czj
    raise ValueError(f"Unsupported J-pole order: {J}")
