"""Numerical equilibrium interpolation utilities."""

from __future__ import annotations

from typing import cast

import numpy as np

from .models import EquilibriumPoint, InterpolationCoefficients, NumericalEquilibrium


def cinterp2d(rg: np.ndarray, zg: np.ndarray, frz: np.ndarray) -> np.ndarray:
    """Pre-compute bilinear interpolation coefficients on a uniform grid."""
    rg = np.asarray(rg, dtype=float).reshape(-1)
    zg = np.asarray(zg, dtype=float).reshape(-1)
    frz = np.asarray(frz, dtype=float)

    nr, nz = frz.shape
    dr = rg[1] - rg[0]
    dz = zg[1] - zg[0]

    fc = np.zeros((4, nr, nz), dtype=float)
    fc[0] = frz
    fc[1, :-1, :] = (fc[0, 1:, :] - fc[0, :-1, :]) / dr
    fc[2, :, :-1] = (fc[0, :, 1:] - fc[0, :, :-1]) / dz
    fc[3, :-1, :-1] = (
        fc[0, 1:, 1:]
        - fc[0, :-1, :-1]
        - fc[1, :-1, :-1] * dr
        - fc[2, :-1, :-1] * dz
    ) / (dr * dz)
    return fc


def fcinterp(eq: NumericalEquilibrium) -> InterpolationCoefficients:
    """Translate the MATLAB `fcinterp.m` preprocessing step."""
    fcns0 = np.zeros((eq.S, 4, eq.nr, eq.nz), dtype=float)
    fcts0 = np.zeros_like(fcns0)
    fcdns0dr = np.zeros_like(fcns0)
    fcdns0dz = np.zeros_like(fcns0)

    for s in range(eq.S):
        fcns0[s] = cinterp2d(eq.rg, eq.zg, eq.fns0[s])
        fcts0[s] = cinterp2d(eq.rg, eq.zg, eq.fts0[s])
        fcdns0dr[s] = cinterp2d(eq.rg, eq.zg, eq.fdns0dr[s])
        fcdns0dz[s] = cinterp2d(eq.rg, eq.zg, eq.fdns0dz[s])

    return InterpolationCoefficients(
        fcB=cinterp2d(eq.rg, eq.zg, eq.fB),
        fcBr=cinterp2d(eq.rg, eq.zg, eq.fBr),
        fcBz=cinterp2d(eq.rg, eq.zg, eq.fBz),
        fcBphi=cinterp2d(eq.rg, eq.zg, eq.fBphi),
        fcpsi=cinterp2d(eq.rg, eq.zg, eq.fpsi),
        fcns0=fcns0,
        fcts0=fcts0,
        fcdBdr=cinterp2d(eq.rg, eq.zg, eq.fdBdr),
        fcdBdz=cinterp2d(eq.rg, eq.zg, eq.fdBdz),
        fcdBrdr=cinterp2d(eq.rg, eq.zg, eq.fdBrdr),
        fcdBrdz=cinterp2d(eq.rg, eq.zg, eq.fdBrdz),
        fcdBzdr=cinterp2d(eq.rg, eq.zg, eq.fdBzdr),
        fcdBzdz=cinterp2d(eq.rg, eq.zg, eq.fdBzdz),
        fcdBphidr=cinterp2d(eq.rg, eq.zg, eq.fdBphidr),
        fcdBphidz=cinterp2d(eq.rg, eq.zg, eq.fdBphidz),
        fcdns0dr=fcdns0dr,
        fcdns0dz=fcdns0dz,
    )


def prepare_numerical_equilibrium(eq: NumericalEquilibrium) -> NumericalEquilibrium:
    if eq.interpolation is not None:
        return eq
    return eq.with_interpolation(fcinterp(eq))


def _bilinear_value(fc: np.ndarray, jr: int, jz: int, hr: float, hz: float) -> float:
    return float(fc[0, jr, jz] + fc[1, jr, jz] * hr + fc[2, jr, jz] * hz + fc[3, jr, jz] * hz * hr)


def sample_numerical_equilibrium(ra: float, za: float, eq: NumericalEquilibrium) -> EquilibriumPoint:
    """Python version of `calpars.m`."""
    eq = prepare_numerical_equilibrium(eq)
    cache = cast(InterpolationCoefficients, eq.interpolation)

    jr = int(np.floor((ra - eq.rg[0]) / eq.dr))
    jz = int(np.floor((za - eq.zg[0]) / eq.dz))

    if jr >= eq.nr - 1 or jr < 0 or jz >= eq.nz - 1 or jz < 0:
        zeros = np.zeros(eq.S, dtype=float)
        return EquilibriumPoint(
            psi=0.0,
            B=0.0,
            Br=0.0,
            Bz=0.0,
            Bphi=0.0,
            ns0=zeros.copy(),
            ts0=zeros.copy(),
            dBdr=0.0,
            dBdz=0.0,
            dBrdr=0.0,
            dBrdz=0.0,
            dBzdr=0.0,
            dBzdz=0.0,
            dBphidr=0.0,
            dBphidz=0.0,
            dns0dr=zeros.copy(),
            dns0dz=zeros.copy(),
            iexit=True,
        )

    hr = ra - eq.rg[jr]
    hz = za - eq.zg[jz]

    ns0 = np.zeros(eq.S, dtype=float)
    ts0 = np.zeros(eq.S, dtype=float)
    dns0dr = np.zeros(eq.S, dtype=float)
    dns0dz = np.zeros(eq.S, dtype=float)
    for s in range(eq.S):
        ns0[s] = _bilinear_value(cache.fcns0[s], jr, jz, hr, hz)
        ts0[s] = _bilinear_value(cache.fcts0[s], jr, jz, hr, hz)
        dns0dr[s] = _bilinear_value(cache.fcdns0dr[s], jr, jz, hr, hz)
        dns0dz[s] = _bilinear_value(cache.fcdns0dz[s], jr, jz, hr, hz)

    return EquilibriumPoint(
        psi=_bilinear_value(cache.fcpsi, jr, jz, hr, hz),
        B=_bilinear_value(cache.fcB, jr, jz, hr, hz),
        Br=_bilinear_value(cache.fcBr, jr, jz, hr, hz),
        Bz=_bilinear_value(cache.fcBz, jr, jz, hr, hz),
        Bphi=_bilinear_value(cache.fcBphi, jr, jz, hr, hz),
        ns0=ns0,
        ts0=ts0,
        dBdr=_bilinear_value(cache.fcdBdr, jr, jz, hr, hz),
        dBdz=_bilinear_value(cache.fcdBdz, jr, jz, hr, hz),
        dBrdr=_bilinear_value(cache.fcdBrdr, jr, jz, hr, hz),
        dBrdz=_bilinear_value(cache.fcdBrdz, jr, jz, hr, hz),
        dBzdr=_bilinear_value(cache.fcdBzdr, jr, jz, hr, hz),
        dBzdz=_bilinear_value(cache.fcdBzdz, jr, jz, hr, hz),
        dBphidr=_bilinear_value(cache.fcdBphidr, jr, jz, hr, hz),
        dBphidz=_bilinear_value(cache.fcdBphidz, jr, jz, hr, hz),
        dns0dr=dns0dr,
        dns0dz=dns0dz,
        iexit=False,
    )
