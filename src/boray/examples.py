"""Prebuilt BORAY cases mirroring the original MATLAB entrypoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import load_mat_equilibrium, load_ray_all
from .models import PowerSettings, RayTraceCase, ReferenceTrajectory
from .solovev import SolovevParameters, create_solovev_equilibrium


def default_matlab_root() -> Path:
    return Path(__file__).resolve().parents[3] / "boray_matlab"


def build_example_case(
    icase: int,
    matlab_root: str | Path | None = None,
    power: PowerSettings | None = None,
    dt0: float | None = None,
    nt0: int | None = None,
) -> RayTraceCase:
    """Build one of the three cases from `boray_main.m`."""
    matlab_root = Path(matlab_root) if matlab_root is not None else default_matlab_root()
    power = power or PowerSettings()

    if icase == 1:
        eqfile = matlab_root / "eqdata" / "genray" / "EAST" / "genray_eqdata.mat"
        ray_all_path = matlab_root / "eqdata" / "genray" / "EAST" / "ray_all.mat"
        ray_all = load_ray_all(ray_all_path)
        yyallray = np.asarray(ray_all["yyallray"], dtype=float)
        frequency_hz = float(np.asarray(ray_all["f"]).squeeze())
        ist = 20
        ray_ids = (1,)
        initial_rays = np.vstack([yyallray[ist, 0:6, rid - 1] for rid in ray_ids])
        reference = ReferenceTrajectory(yyallray=yyallray, ist=ist, ray_ids=ray_ids)
        return RayTraceCase(
            equilibrium=load_mat_equilibrium(eqfile),
            frequency_hz=frequency_hz,
            initial_rays=initial_rays,
            dt0=0.0003 if dt0 is None else dt0,
            nt0=5000 if nt0 is None else nt0,
            save_directory=matlab_root / "output" / "genray" / "EAST",
            power=power,
            solver_rescale=2.0,
            label="icase=1",
            reference=reference,
        )

    if icase == 2:
        eqfile = matlab_root / "eqdata" / "mirror" / "zjs_eqdata.mat"
        initial_rays = np.array(
            [
                [0.125, 0.0, -0.2, 900.0, 0.0, -16.3],
                [0.125, 0.0, 0.0, 900.0, 0.0, -16.3],
                [0.125, 0.0, 0.2, 900.0, 0.0, -16.3],
                [0.125, 0.0, 0.4, 900.0, 0.0, -16.3],
            ],
            dtype=float,
        )
        return RayTraceCase(
            equilibrium=load_mat_equilibrium(eqfile),
            frequency_hz=160e6,
            initial_rays=initial_rays,
            dt0=0.01 if dt0 is None else dt0,
            nt0=4000 if nt0 is None else nt0,
            save_directory=matlab_root / "output" / "zjs_mirror",
            power=power,
            solver_rescale=2.0,
            label="icase=2",
        )

    if icase == 3:
        equilibrium = create_solovev_equilibrium(SolovevParameters())
        initial_rays = np.array([[0.85, 0.0, -0.2, -5.4910, -1.6, 0.8]], dtype=float)
        return RayTraceCase(
            equilibrium=equilibrium,
            frequency_hz=4.5e6,
            initial_rays=initial_rays,
            dt0=0.02 if dt0 is None else dt0,
            nt0=4000 if nt0 is None else nt0,
            save_directory=matlab_root / "output" / "analytical",
            power=power,
            solver_rescale=2.0,
            label="icase=3",
        )

    raise ValueError(f"Unsupported icase: {icase}")
