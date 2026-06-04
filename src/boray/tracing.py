"""Ray-tracing driver for the BORAY Python port."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C, C2, TWOPI
from .dispersion import colddr, dydt, hotdreach, solve_initial_kr
from .equilibrium import prepare_numerical_equilibrium, sample_numerical_equilibrium
from .models import CaseResult, NumericalEquilibrium, PowerAbsorptionResult, PowerSettings, RayTraceCase, RayTrajectory, SingleRayResult
from .solovev import SolovevEquilibrium, sample_solovev_equilibrium


def _sample_equilibrium(equilibrium: NumericalEquilibrium | SolovevEquilibrium, r: float, z: float):
    if isinstance(equilibrium, NumericalEquilibrium):
        return sample_numerical_equilibrium(r, z, equilibrium)
    return sample_solovev_equilibrium(r, z, equilibrium)


def _species_count(equilibrium: NumericalEquilibrium | SolovevEquilibrium) -> int:
    return int(equilibrium.S)


def _qs_ms(equilibrium: NumericalEquilibrium | SolovevEquilibrium):
    return np.asarray(equilibrium.qs, dtype=float), np.asarray(equilibrium.ms, dtype=float)


def _grid_r_bounds(equilibrium: NumericalEquilibrium | SolovevEquilibrium):
    return float(np.min(equilibrium.rg)), float(np.max(equilibrium.rg))


def _trim_legacy(legacy: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(legacy[:, 6])
    return legacy[valid]


def _rk4_trace(
    equilibrium: NumericalEquilibrium | SolovevEquilibrium,
    initial_state: np.ndarray,
    traced_kr: float,
    frequency_hz: float,
    dt0: float,
    nt0: int,
    solver_rescale: float,
) -> RayTrajectory:
    qs, ms = _qs_ms(equilibrium)
    S = _species_count(equilibrium)
    nt = int(nt0 * solver_rescale)
    dt = dt0 / C / solver_rescale
    w = TWOPI * frequency_hz
    legacy = np.full((nt, 18 + 2 * S), np.nan, dtype=float)

    r, phi, z, _kr_guess, nphi, kz = initial_state.astype(float)
    kr = float(traced_kr)
    rmin, _rmax = _grid_r_bounds(equilibrium)

    for it in range(nt):
        legacy[it, 0:6] = [r, phi, z, kr, nphi, kz]
        point1 = _sample_equilibrium(equilibrium, r, z)
        if point1.iexit:
            break

        kpar1, kper21, dFdr1, dFdphi1, dFdz1, dFdkr1, dFdnphi1, dFdkz1, dFdw1, D1 = dydt(
            r, phi, z, kr, nphi, kz, point1, qs, ms, w, C2
        )

        rtmp = r - dFdkr1 / dFdw1 * dt
        phitmp = phi - dFdnphi1 / dFdw1 * dt
        ztmp = z - dFdkz1 / dFdw1 * dt
        krtmp = kr + dFdr1 / dFdw1 * dt
        nphitmp = nphi + dFdphi1 / dFdw1 * dt
        kztmp = kz + dFdz1 / dFdw1 * dt

        point2 = _sample_equilibrium(equilibrium, rtmp, ztmp)
        if point2.iexit:
            break
        kpar2, kper22, dFdr2, dFdphi2, dFdz2, dFdkr2, dFdnphi2, dFdkz2, dFdw2, D2 = dydt(
            rtmp, phitmp, ztmp, krtmp, nphitmp, kztmp, point2, qs, ms, w, C2
        )

        rtmp = r - dFdkr2 / dFdw2 * dt
        phitmp = phi - dFdnphi2 / dFdw2 * dt
        ztmp = z - dFdkz2 / dFdw2 * dt
        krtmp = kr + dFdr2 / dFdw2 * dt
        nphitmp = nphi + dFdphi2 / dFdw2 * dt
        kztmp = kz + dFdz2 / dFdw2 * dt

        point3 = _sample_equilibrium(equilibrium, rtmp, ztmp)
        if point3.iexit:
            break
        kpar3, kper23, dFdr3, dFdphi3, dFdz3, dFdkr3, dFdnphi3, dFdkz3, dFdw3, D3 = dydt(
            rtmp, phitmp, ztmp, krtmp, nphitmp, kztmp, point3, qs, ms, w, C2
        )

        rtmp = r - dFdkr3 / dFdw3 * dt
        phitmp = phi - dFdnphi3 / dFdw3 * dt
        ztmp = z - dFdkz3 / dFdw3 * dt
        krtmp = kr + dFdr3 / dFdw3 * dt
        nphitmp = nphi + dFdphi3 / dFdw3 * dt
        kztmp = kz + dFdz3 / dFdw3 * dt

        point4 = _sample_equilibrium(equilibrium, rtmp, ztmp)
        if point4.iexit:
            break
        kpar4, kper24, dFdr4, dFdphi4, dFdz4, dFdkr4, dFdnphi4, dFdkz4, dFdw4, D4 = dydt(
            rtmp, phitmp, ztmp, krtmp, nphitmp, kztmp, point4, qs, ms, w, C2
        )

        drdt = -(dFdkr1 / dFdw1 + 2.0 * dFdkr2 / dFdw2 + 2.0 * dFdkr3 / dFdw3 + dFdkr4 / dFdw4) / 6.0
        dphidt = -(dFdnphi1 / dFdw1 + 2.0 * dFdnphi2 / dFdw2 + 2.0 * dFdnphi3 / dFdw3 + dFdnphi4 / dFdw4) / 6.0
        dzdt = -(dFdkz1 / dFdw1 + 2.0 * dFdkz2 / dFdw2 + 2.0 * dFdkz3 / dFdw3 + dFdkz4 / dFdw4) / 6.0
        dkrdt = (dFdr1 / dFdw1 + 2.0 * dFdr2 / dFdw2 + 2.0 * dFdr3 / dFdw3 + dFdr4 / dFdw4) / 6.0
        dnphidt = (dFdphi1 / dFdw1 + 2.0 * dFdphi2 / dFdw2 + 2.0 * dFdphi3 / dFdw3 + dFdphi4 / dFdw4) / 6.0
        dkzdt = (dFdz1 / dFdw1 + 2.0 * dFdz2 / dFdw2 + 2.0 * dFdz3 / dFdw3 + dFdz4 / dFdw4) / 6.0

        r = r + drdt * dt
        phi = phi + dphidt * dt
        z = z + dzdt * dt
        kr = kr + dkrdt * dt
        nphi = nphi + dnphidt * dt
        kz = kz + dkzdt * dt

        if r < rmin:
            kr = -abs(kr)

        legacy[it, 6] = (it + 1) * dt
        legacy[it, 7] = np.real_if_close(D1)
        legacy[it, 8] = drdt
        legacy[it, 9] = dphidt
        legacy[it, 10] = dzdt
        legacy[it, 11] = dkrdt
        legacy[it, 12] = dnphidt
        legacy[it, 13] = dkzdt
        legacy[it, 14] = np.real_if_close(kpar1)
        legacy[it, 15] = np.sqrt(np.real_if_close(kper21))
        legacy[it, 16] = point4.B
        for s in range(S):
            legacy[it, 17 + 2 * s] = point4.ns0[s]
            legacy[it, 18 + 2 * s] = point4.ts0[s]
        legacy[it, -1] = point4.psi

    return RayTrajectory(legacy_matrix=_trim_legacy(legacy))


def _power_from_trajectory(trajectory: RayTrajectory, frequency_hz: float, power: PowerSettings, qs: np.ndarray, ms: np.ndarray) -> PowerAbsorptionResult | None:
    legacy = trajectory.legacy_matrix
    if not power.enabled or legacy.shape[0] < 2:
        return None

    ntp0 = 200
    djp = max(1, legacy.shape[0] // ntp0)
    yyp = legacy[::djp]
    if yyp.shape[0] < 2:
        yyp = legacy

    S = (legacy.shape[1] - 18) // 2
    rp = yyp[:, 0]
    zp = yyp[:, 2]
    tp = yyp[:, 6]
    ns0 = np.zeros((yyp.shape[0], S), dtype=float)
    Ts0 = np.zeros_like(ns0)
    for s in range(S):
        ns0[:, s] = yyp[:, 17 + 2 * s]
        Ts0[:, s] = yyp[:, 18 + 2 * s]

    B0 = yyp[:, 16]
    kz = np.abs(yyp[:, 14])
    kx = np.abs(yyp[:, 15])
    wt = np.full(tp.shape, frequency_hz * TWOPI, dtype=float)

    ww, detD = colddr(qs, ms, ns0, B0, kx, kz, wt)
    del detD
    ww_each, detD_each, N_each = hotdreach(qs, ms, ns0, Ts0, B0, kx, kz, ww, power.jeach, power.joutw, power.N, power.J)
    wwh = ww_each.copy()

    if power.joutw == 0:
        ww_each2, detD_each2, _ = hotdreach(qs, ms, ns0, Ts0, B0, kx, kz, ww * 0.99999, power.jeach, power.joutw, power.N, power.J)
        wi_each2 = np.zeros_like(ww_each2)
        for jp in range(len(ww)):
            wi_each2[jp, :] = np.imag(detD_each[jp, :]) / (np.real(detD_each2[jp, :] - detD_each[jp, :]) / (0.00001 * ww[jp]))
        wwh = np.real(ww_each) + 1j * wi_each2

    dtp = tp[1] - tp[0]
    Ptp = np.zeros((len(tp), N_each), dtype=float)
    for js in range(N_each):
        damp = 2.0 * np.cumsum(np.imag(wwh[:, js]) * dtp)
        Ptp[:, js] = 1.0 - np.exp(damp)

    return PowerAbsorptionResult(
        sample_time=tp,
        sample_r=rp,
        sample_z=zp,
        cold_omega=np.asarray(ww, dtype=complex),
        hot_omega=np.asarray(wwh, dtype=complex),
        absorbed_power=Ptp,
        species_count=N_each,
    )


def run_case(case: RayTraceCase) -> CaseResult:
    """Run a BORAY case for one or more rays."""
    equilibrium = case.equilibrium
    if isinstance(equilibrium, NumericalEquilibrium):
        equilibrium = prepare_numerical_equilibrium(equilibrium)

    qs, ms = _qs_ms(equilibrium)
    initial_rays = np.atleast_2d(np.asarray(case.initial_rays, dtype=float))
    results: list[SingleRayResult] = []
    w = TWOPI * case.frequency_hz

    for row in initial_rays:
        r, phi, z, kr_guess, nphi, kz = row
        point = _sample_equilibrium(equilibrium, float(r), float(z))
        if point.iexit:
            raise ValueError("Initial ray is out of the equilibrium range.")
        traced_kr = solve_initial_kr(float(r), point, float(nphi), float(kz), w, qs, ms, float(kr_guess))
        trajectory = _rk4_trace(equilibrium, row, traced_kr, case.frequency_hz, case.dt0, case.nt0, case.solver_rescale)
        power_result = _power_from_trajectory(trajectory, case.frequency_hz, case.power, qs, ms)
        results.append(SingleRayResult(initial_state=row.astype(float), traced_kr=traced_kr, trajectory=trajectory, power=power_result))

    return CaseResult(case=case, rays=tuple(results))
