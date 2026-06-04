"""Analytical Solovev equilibrium support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .constants import EV_TO_J, ME, QE
from .models import EquilibriumPoint


@dataclass(frozen=True)
class SolovevParameters:
    iconfig: int = 1
    R0: float | complex = 0.64
    B0: float = 0.32
    q0: float = 1.58
    Rx: float = 0.17
    E: float = 1.5
    tau: float = 0.8
    qs0: np.ndarray = field(default_factory=lambda: np.array([-1.0, 1.0, 2.0], dtype=float))
    ms0: np.ndarray = field(default_factory=lambda: np.array([1.0, 1836.0, 4.0 * 1836.0], dtype=float))
    ns00: np.ndarray = field(default_factory=lambda: np.array([5.5e18, 4.95e18, 2.75e17], dtype=float))
    ts00: np.ndarray = field(default_factory=lambda: np.array([200.0, 50.0, 50.0], dtype=float))
    Lns: np.ndarray = field(default_factory=lambda: np.array([0.9, 0.9, 0.9], dtype=float))
    Lts: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8], dtype=float))


@dataclass(frozen=True)
class SolovevEquilibrium:
    params: SolovevParameters
    qs: np.ndarray
    ms: np.ndarray
    S: int
    rg: np.ndarray
    zg: np.ndarray
    rr: np.ndarray
    zz: np.ndarray
    fpsi: np.ndarray
    fB: np.ndarray
    fBr: np.ndarray
    fBz: np.ndarray
    fBphi: np.ndarray
    fns0: np.ndarray
    fts0: np.ndarray
    fdBdr: np.ndarray
    fdBdz: np.ndarray
    fdBrdr: np.ndarray
    fdBrdz: np.ndarray
    fdBzdr: np.ndarray
    fdBzdz: np.ndarray
    fdBphidr: np.ndarray
    fdBphidz: np.ndarray
    fdns0dr: np.ndarray
    fdns0dz: np.ndarray
    n0: float
    psix: float
    ffpsi: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffB: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffBr: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffBz: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffBphi: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffns0: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray]
    ffts0: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBdr: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBdz: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBrdr: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBrdz: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBzdr: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBzdz: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBphidr: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdBphidz: Callable[[np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdns0dr: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray]
    ffdns0dz: Callable[[int, np.ndarray | float, np.ndarray | float], np.ndarray]


def create_solovev_equilibrium(params: SolovevParameters) -> SolovevEquilibrium:
    """Build the analytical equilibrium used by `initialsolovev.m`."""
    iconfig = params.iconfig
    R0 = params.R0
    B0 = params.B0
    q0 = params.q0
    Rx = params.Rx
    E = params.E
    tau = params.tau
    Lns = np.asarray(params.Lns, dtype=float)
    Lts = np.asarray(params.Lts, dtype=float)

    if iconfig == 1:
        kappa = 2 * E / np.sqrt(1 - Rx**2 / (R0**2))
        psi0 = B0 * R0**2 / (8 * q0)
        rmin = 0.8 * Rx
        rmax = 2.0 * R0
        zmax = R0 * kappa * 0.8
        zmin = -zmax
    elif iconfig == 2:
        Rx = 0.0
        tau = 0.0
        kappa = 2 * E / np.sqrt(1 - Rx**2 / (R0**2))
        psi0 = B0 * R0**2 / 4
        rmin = 0.0
        rmax = 2.0 * R0
        zmax = R0 * kappa
        zmin = -zmax
    else:
        Rx = 0.0
        tau = 0.0
        kappa = 2 * E / np.sqrt(1 - Rx**2 / (R0**2))
        psi0 = B0 * abs(R0**2) / 4
        rmin = 0.0
        rmax = 2.0 * abs(R0)
        zmax = abs(R0) * kappa
        zmin = -zmax

    zx_arg = tau * R0**2 * np.log(Rx**2 / (R0**2) + 1e-10) + (2 + tau) * (R0**2 - Rx**2)
    Zx = np.abs(E * np.sqrt(zx_arg))

    def ffpsi(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        R = np.asarray(R)
        Z = np.asarray(Z)
        return (
            psi0
            / R0**4
            * (
                (R**2 - R0**2) ** 2
                + Z**2 / E**2 * (R**2 - Rx**2)
                - tau
                * R0**2
                * (
                    R**2 * np.log((R + 1e-10) ** 2 / R0**2)
                    - (R**2 - R0**2)
                    - (R**2 - R0**2) ** 2 / (2 * R0**2)
                )
            )
        )

    def ffBr(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        R = np.asarray(R)
        Z = np.asarray(Z)
        return -2 * psi0 / (R + 1e-10) / R0**4 * (Z / E**2 * (R**2 - Rx**2))

    def ffBz(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        R = np.asarray(R)
        Z = np.asarray(Z)
        return 2 * psi0 / R0**4 * (
            2 * (R**2 - R0**2)
            + Z**2 / E**2
            - tau * R0**2 * (np.log((R + 1e-10) ** 2 / R0**2) - (R**2 - R0**2) / R0**2)
        )

    if iconfig == 1:
        ffBphi = lambda R, Z: B0 * R0 / (np.asarray(R) + 1e-10)
        ffdBphidr = lambda R, Z: -B0 * R0 / (np.asarray(R) + 1e-10) ** 2
    else:
        ffBphi = lambda R, Z: np.zeros_like(np.asarray(R), dtype=float)
        ffdBphidr = lambda R, Z: np.zeros_like(np.asarray(R), dtype=float)

    ffdBphidz = lambda R, Z: np.zeros_like(np.asarray(R), dtype=float)
    ffdBrdr = lambda R, Z: -2 * psi0 / R0**4 * (np.asarray(Z) / E**2 * (1 + Rx**2 / np.asarray(R) ** 2))
    ffdBrdz = lambda R, Z: -2 * psi0 / (np.asarray(R) + 1e-10) / R0**4 * (1 / E**2 * (np.asarray(R) ** 2 - Rx**2))
    ffdBzdr = lambda R, Z: 4 * psi0 / R0**4 * np.asarray(R) * (2 + tau - tau * R0**2 / (np.asarray(R) + 1e-10) ** 2)
    ffdBzdz = lambda R, Z: 4 * psi0 / R0**4 * (np.asarray(Z) / E**2)

    def ffB(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return np.sqrt(ffBr(R, Z) ** 2 + ffBz(R, Z) ** 2 + ffBphi(R, Z) ** 2)

    def ffdBdr(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return (
            ffBr(R, Z) * ffdBrdr(R, Z) + ffBz(R, Z) * ffdBzdr(R, Z) + ffBphi(R, Z) * ffdBphidr(R, Z)
        ) / ffB(R, Z)

    def ffdBdz(R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return (
            ffBr(R, Z) * ffdBrdz(R, Z) + ffBz(R, Z) * ffdBzdz(R, Z) + ffBphi(R, Z) * ffdBphidz(R, Z)
        ) / ffB(R, Z)

    Zx = np.nan_to_num(np.asarray(Zx), nan=0.0)
    psix = float(np.real_if_close(ffpsi(Rx, Zx)))

    qs = np.asarray(params.qs0, dtype=float) * QE
    ms = np.asarray(params.ms0, dtype=float) * ME
    ns00 = np.asarray(params.ns00, dtype=float)
    ts00 = np.asarray(params.ts00, dtype=float)
    S = len(qs)

    def ffns0(s: int, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return ns00[s] * np.exp(-ffpsi(R, Z) / (psix * Lns[s] ** 2))

    def ffts0(s: int, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return ts00[s] * np.exp(-ffpsi(R, Z) / (psix * Lts[s] ** 2))

    def ffdns0dpsi(s: int, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return -ffns0(s, R, Z) / (psix * Lns[s] ** 2)

    def ffdns0dr(s: int, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return np.asarray(R) * ffBz(R, Z) * ffdns0dpsi(s, R, Z)

    def ffdns0dz(s: int, R: np.ndarray | float, Z: np.ndarray | float) -> np.ndarray:
        return -np.asarray(R) * ffBr(R, Z) * ffdns0dpsi(s, R, Z)

    dr = 0.02 * (rmax - rmin)
    dz = 0.02 * (zmax - zmin)
    rg = np.arange(rmin, rmax + 0.5 * dr, dr, dtype=float)
    zg = np.arange(zmin, zmax + 0.5 * dz, dz, dtype=float)
    rr, zz = np.meshgrid(rg, zg, indexing="ij")

    fpsi = ffpsi(rr, zz)
    fB = ffB(rr, zz)
    fBr = ffBr(rr, zz)
    fBz = ffBz(rr, zz)
    fBphi = ffBphi(rr, zz)
    fdBdr = ffdBdr(rr, zz)
    fdBdz = ffdBdz(rr, zz)
    fdBrdr = ffdBrdr(rr, zz)
    fdBrdz = ffdBrdz(rr, zz)
    fdBzdr = ffdBzdr(rr, zz)
    fdBzdz = ffdBzdz(rr, zz)
    fdBphidr = ffdBphidr(rr, zz)
    fdBphidz = ffdBphidz(rr, zz)
    fns0 = np.stack([ffns0(s, rr, zz) for s in range(S)], axis=0)
    fts0 = np.stack([ffts0(s, rr, zz) for s in range(S)], axis=0)
    fdns0dr = np.stack([ffdns0dr(s, rr, zz) for s in range(S)], axis=0)
    fdns0dz = np.stack([ffdns0dz(s, rr, zz) for s in range(S)], axis=0)

    n0 = float(np.max(fns0[0]))
    return SolovevEquilibrium(
        params=params,
        qs=qs,
        ms=ms,
        S=S,
        rg=rg,
        zg=zg,
        rr=rr,
        zz=zz,
        fpsi=np.asarray(np.real_if_close(fpsi), dtype=float),
        fB=np.asarray(np.real_if_close(fB), dtype=float),
        fBr=np.asarray(np.real_if_close(fBr), dtype=float),
        fBz=np.asarray(np.real_if_close(fBz), dtype=float),
        fBphi=np.asarray(np.real_if_close(fBphi), dtype=float),
        fns0=np.asarray(np.real_if_close(fns0), dtype=float),
        fts0=np.asarray(np.real_if_close(fts0), dtype=float),
        fdBdr=np.asarray(np.real_if_close(fdBdr), dtype=float),
        fdBdz=np.asarray(np.real_if_close(fdBdz), dtype=float),
        fdBrdr=np.asarray(np.real_if_close(fdBrdr), dtype=float),
        fdBrdz=np.asarray(np.real_if_close(fdBrdz), dtype=float),
        fdBzdr=np.asarray(np.real_if_close(fdBzdr), dtype=float),
        fdBzdz=np.asarray(np.real_if_close(fdBzdz), dtype=float),
        fdBphidr=np.asarray(np.real_if_close(fdBphidr), dtype=float),
        fdBphidz=np.asarray(np.real_if_close(fdBphidz), dtype=float),
        fdns0dr=np.asarray(np.real_if_close(fdns0dr), dtype=float),
        fdns0dz=np.asarray(np.real_if_close(fdns0dz), dtype=float),
        n0=n0,
        psix=psix,
        ffpsi=ffpsi,
        ffB=ffB,
        ffBr=ffBr,
        ffBz=ffBz,
        ffBphi=ffBphi,
        ffns0=ffns0,
        ffts0=ffts0,
        ffdBdr=ffdBdr,
        ffdBdz=ffdBdz,
        ffdBrdr=ffdBrdr,
        ffdBrdz=ffdBrdz,
        ffdBzdr=ffdBzdr,
        ffdBzdz=ffdBzdz,
        ffdBphidr=ffdBphidr,
        ffdBphidz=ffdBphidz,
        ffdns0dr=ffdns0dr,
        ffdns0dz=ffdns0dz,
    )


def sample_solovev_equilibrium(ra: float, za: float, eq: SolovevEquilibrium) -> EquilibriumPoint:
    """Python version of `calpars_solovev.m`."""
    ns0 = np.array([eq.ffns0(s, ra, za) for s in range(eq.S)], dtype=float)
    ts0 = np.array([eq.ffts0(s, ra, za) for s in range(eq.S)], dtype=float)
    dns0dr = np.array([eq.ffdns0dr(s, ra, za) for s in range(eq.S)], dtype=float)
    dns0dz = np.array([eq.ffdns0dz(s, ra, za) for s in range(eq.S)], dtype=float)
    return EquilibriumPoint(
        psi=float(np.real_if_close(eq.ffpsi(ra, za))),
        B=float(np.real_if_close(eq.ffB(ra, za))),
        Br=float(np.real_if_close(eq.ffBr(ra, za))),
        Bz=float(np.real_if_close(eq.ffBz(ra, za))),
        Bphi=float(np.real_if_close(eq.ffBphi(ra, za))),
        ns0=ns0,
        ts0=ts0,
        dBdr=float(np.real_if_close(eq.ffdBdr(ra, za))),
        dBdz=float(np.real_if_close(eq.ffdBdz(ra, za))),
        dBrdr=float(np.real_if_close(eq.ffdBrdr(ra, za))),
        dBrdz=float(np.real_if_close(eq.ffdBrdz(ra, za))),
        dBzdr=float(np.real_if_close(eq.ffdBzdr(ra, za))),
        dBzdz=float(np.real_if_close(eq.ffdBzdz(ra, za))),
        dBphidr=float(np.real_if_close(eq.ffdBphidr(ra, za))),
        dBphidz=float(np.real_if_close(eq.ffdBphidz(ra, za))),
        dns0dr=dns0dr,
        dns0dz=dns0dz,
        iexit=False,
    )
