"""Input/output helpers for BORAY MATLAB data products."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import io as scipy_io
from scipy.interpolate import interp1d

from .equilibrium import prepare_numerical_equilibrium
from .models import NumericalEquilibrium


@dataclass(frozen=True)
class GFileData:
    nr: int
    nz: int
    R: np.ndarray
    Z: np.ndarray
    RR: np.ndarray
    ZZ: np.ndarray
    psi: np.ndarray
    Br: np.ndarray
    Bz: np.ndarray
    Bphi: np.ndarray
    pressure: np.ndarray
    q: np.ndarray
    Psi_axis: float
    Psi_bound: float
    Raxis: float
    Zaxis: float
    B0: float
    Rbound: np.ndarray
    Zbound: np.ndarray
    Rlimiter: np.ndarray
    Zlimiter: np.ndarray


def _squeeze_float(data: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=float).squeeze()


def load_mat_equilibrium(path: str | Path) -> NumericalEquilibrium:
    """Load a legacy BORAY `.mat` equilibrium file."""
    path = Path(path)
    data = scipy_io.loadmat(path, squeeze_me=True)

    rg = _squeeze_float(data, "rg").reshape(-1)
    zg = _squeeze_float(data, "zg").reshape(-1)
    rr = _squeeze_float(data, "rr")
    zz = _squeeze_float(data, "zz")

    eq = NumericalEquilibrium(
        rg=rg,
        zg=zg,
        dr=float(np.asarray(data["dr"]).squeeze()),
        dz=float(np.asarray(data["dz"]).squeeze()),
        rr=rr,
        zz=zz,
        fB=_squeeze_float(data, "fB"),
        fBr=_squeeze_float(data, "fBr"),
        fBz=_squeeze_float(data, "fBz"),
        fBphi=_squeeze_float(data, "fBphi"),
        fns0=np.asarray(data["fns0"], dtype=float),
        fts0=np.asarray(data["fts0"], dtype=float),
        fdBdr=_squeeze_float(data, "fdBdr"),
        fdBdz=_squeeze_float(data, "fdBdz"),
        fdBrdr=_squeeze_float(data, "fdBrdr"),
        fdBrdz=_squeeze_float(data, "fdBrdz"),
        fdBzdr=_squeeze_float(data, "fdBzdr"),
        fdBzdz=_squeeze_float(data, "fdBzdz"),
        fdBphidr=_squeeze_float(data, "fdBphidr"),
        fdBphidz=_squeeze_float(data, "fdBphidz"),
        fdns0dr=np.asarray(data["fdns0dr"], dtype=float),
        fdns0dz=np.asarray(data["fdns0dz"], dtype=float),
        fpsi=_squeeze_float(data, "fpsi"),
        qs=_squeeze_float(data, "qs").reshape(-1),
        ms=_squeeze_float(data, "ms").reshape(-1),
        S=int(np.asarray(data["S"]).squeeze()),
        R0=float(np.asarray(data["R0"]).squeeze()) if "R0" in data else None,
        Z0=float(np.asarray(data["Z0"]).squeeze()) if "Z0" in data else None,
        B0=float(np.asarray(data["B0"]).squeeze()) if "B0" in data else None,
        n0=float(np.asarray(data["n0"]).squeeze()) if "n0" in data else None,
        psilim=float(np.asarray(data["psilim"]).squeeze()) if "psilim" in data else None,
    )
    return prepare_numerical_equilibrium(eq)


def load_ray_all(path: str | Path) -> dict[str, Any]:
    """Load the MATLAB `ray_all.mat` helper file."""
    path = Path(path)
    data = scipy_io.loadmat(path, squeeze_me=True)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def read_genray_netcdf(path: str | Path) -> dict[str, np.ndarray]:
    """Port of `read_genray.m`."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("read_genray_netcdf requires the optional 'netCDF4' dependency.") from exc

    path = Path(path)
    with Dataset(path) as ds:
        def read(name: str) -> np.ndarray:
            return np.asarray(ds.variables[name][:])

        c = 2.99792458e8
        out = {
            "wr": read("wr") / 100.0,
            "wz": read("wz") / 100.0,
            "wphi": read("wphi"),
            "wn_r": read("wn_r"),
            "wn_z": read("wn_z"),
            "wn_phi": read("wn_phi"),
            "w_theta_pol": read("w_theta_pol") * np.pi / 180.0,
            "wnpar": read("wnpar"),
            "wnper": read("wnper"),
            "delpwr": read("delpwr") * 1e-7,
            "te": read("ste") * 1e3,
            "ne": read("sene") * 1e6,
            "br": read("sb_r") * 1e-4,
            "bz": read("sb_z") * 1e-4,
            "bphi": read("sb_phi") * 1e-4,
            "btot": read("sbtot") * 1e-4,
            "vgr_r": read("vgr_r") * c,
            "vgr_z": read("vgr_z") * c,
            "vgr_phi": read("vgr_phi") * c,
            "flux_r": read("flux_r") * c,
            "flux_z": read("flux_z") * c,
            "flux_phi": read("flux_phi") * c,
            "freqcy": read("freqcy"),
            "dmas": read("dmas"),
            "charge": read("charge"),
            "eqdsk_r": read("eqdsk_r"),
            "eqdsk_z": read("eqdsk_z"),
            "eqdsk_psi": read("eqdsk_psi"),
            "indexrho": read("indexrho"),
            "psifactr": read("psifactr"),
            "binvol": read("binvol"),
            "binarea": read("binarea"),
            "rho_bin": read("rho_bin"),
            "rho_bin_center": read("rho_bin_center"),
            "densprof": read("densprof") * 1e6,
            "temprof": read("temprof") * 1e3,
            "zefprof": read("zefprof"),
            "w_r_densprof_nc": read("w_r_densprof_nc"),
            "w_dens_vs_r_nc": read("w_dens_vs_r_nc"),
            "w_temp_vs_r_nc": read("w_temp_vs_r_nc") * 1e3,
            "w_zeff_vs_r_nc": read("w_zeff_vs_r_nc"),
            "iabsorp": read("iabsorp"),
        }

    f = out["freqcy"]
    w = 2.0 * np.pi * f
    out["wkr"] = out["wn_r"] / c * w
    out["wkz"] = out["wn_z"] / c * w
    out["wkphi"] = out["wn_phi"] / c * w
    return out


def read_gfile(path: str | Path) -> GFileData:
    """Parse an EFIT gfile similarly to the MATLAB helper."""
    path = Path(path)
    tokens = path.read_text().split()
    i = 0

    def next_str() -> str:
        nonlocal i
        value = tokens[i]
        i += 1
        return value

    def next_int() -> int:
        return int(next_str())

    def next_float() -> float:
        return float(next_str())

    _ = [next_str() for _ in range(5)]
    _n3 = next_int()
    nr = next_int()
    nz = next_int()

    Rboxlen = next_float()
    Zboxlen = next_float()
    R0 = next_float()
    Rmin = next_float()
    Z0 = next_float()

    Raxis = next_float()
    Zaxis = next_float()
    Psi_axis = next_float()
    Psi_bound = next_float()
    B0 = next_float()

    _current = next_float()
    _Psi_axis1 = next_float()
    _xdum1 = next_float()
    Raxis = next_float()
    _xdum2 = next_float()
    Zaxis = next_float()
    _xdum3 = next_float()
    Psi_bound = next_float()
    _xdum4 = next_float()
    _xdum5 = next_float()

    f = np.array([next_float() for _ in range(nr)], dtype=float)
    pressure = np.array([next_float() for _ in range(nr)], dtype=float)
    _ffprime = np.array([next_float() for _ in range(nr)], dtype=float)
    _pprime = np.array([next_float() for _ in range(nr)], dtype=float)
    psi = np.array([next_float() for _ in range(nr * nz)], dtype=float).reshape(nr, nz)
    q = np.array([next_float() for _ in range(nr)], dtype=float)

    R = np.linspace(Rmin, Rmin + Rboxlen, nr)
    Z = np.linspace(Z0 - Zboxlen * 0.5, Z0 + Zboxlen * 0.5, nz)
    dR = R[1] - R[0]
    dZ = Z[1] - Z[0]

    RR = np.repeat(R[:, None], nz, axis=1)
    ZZ = np.repeat(Z[None, :], nr, axis=0)
    dpsi_dR, dpsi_dZ = np.gradient(psi, dR, dZ, edge_order=2)
    Br = -dpsi_dZ / RR
    Bz = dpsi_dR / RR

    if Psi_axis > Psi_bound:
        psi_1d = np.linspace(Psi_bound, Psi_axis, nr)
    else:
        psi_1d = np.linspace(Psi_axis, Psi_bound, nr)
    g = interp1d(psi_1d, f, kind="cubic", fill_value="extrapolate")
    gfun = g(psi)
    Bphi = gfun / RR

    nbound = next_int()
    nlimiter = next_int()
    RZbound = np.array([next_float() for _ in range(nbound * 2)], dtype=float).reshape(2, nbound)
    Rbound = RZbound[0]
    Zbound = RZbound[1]

    _nbound2 = next_int()
    RZlimiter = np.array([next_float() for _ in range(nlimiter * 2)], dtype=float).reshape(2, nlimiter)
    Rlimiter = RZlimiter[0]
    Zlimiter = RZlimiter[1]

    return GFileData(
        nr=nr,
        nz=nz,
        R=R,
        Z=Z,
        RR=RR,
        ZZ=ZZ,
        psi=psi,
        Br=Br,
        Bz=Bz,
        Bphi=Bphi,
        pressure=pressure,
        q=q,
        Psi_axis=Psi_axis,
        Psi_bound=Psi_bound,
        Raxis=Raxis,
        Zaxis=Zaxis,
        B0=B0,
        Rbound=Rbound,
        Zbound=Zbound,
        Rlimiter=Rlimiter,
        Zlimiter=Zlimiter,
    )
