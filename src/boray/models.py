"""Shared dataclasses for the BORAY port."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class InterpolationCoefficients:
    fcB: FloatArray
    fcBr: FloatArray
    fcBz: FloatArray
    fcBphi: FloatArray
    fcpsi: FloatArray
    fcns0: FloatArray
    fcts0: FloatArray
    fcdBdr: FloatArray
    fcdBdz: FloatArray
    fcdBrdr: FloatArray
    fcdBrdz: FloatArray
    fcdBzdr: FloatArray
    fcdBzdz: FloatArray
    fcdBphidr: FloatArray
    fcdBphidz: FloatArray
    fcdns0dr: FloatArray
    fcdns0dz: FloatArray


@dataclass(frozen=True)
class NumericalEquilibrium:
    rg: FloatArray
    zg: FloatArray
    dr: float
    dz: float
    rr: FloatArray
    zz: FloatArray
    fB: FloatArray
    fBr: FloatArray
    fBz: FloatArray
    fBphi: FloatArray
    fns0: FloatArray
    fts0: FloatArray
    fdBdr: FloatArray
    fdBdz: FloatArray
    fdBrdr: FloatArray
    fdBrdz: FloatArray
    fdBzdr: FloatArray
    fdBzdz: FloatArray
    fdBphidr: FloatArray
    fdBphidz: FloatArray
    fdns0dr: FloatArray
    fdns0dz: FloatArray
    fpsi: FloatArray
    qs: FloatArray
    ms: FloatArray
    S: int
    R0: float | complex | None = None
    Z0: float | complex | None = None
    B0: float | None = None
    n0: float | None = None
    psilim: float | None = None
    interpolation: InterpolationCoefficients | None = None

    @property
    def nr(self) -> int:
        return int(self.fB.shape[0])

    @property
    def nz(self) -> int:
        return int(self.fB.shape[1])

    def with_interpolation(self, interpolation: InterpolationCoefficients) -> "NumericalEquilibrium":
        return replace(self, interpolation=interpolation)


@dataclass(frozen=True)
class EquilibriumPoint:
    psi: float
    B: float
    Br: float
    Bz: float
    Bphi: float
    ns0: FloatArray
    ts0: FloatArray
    dBdr: float
    dBdz: float
    dBrdr: float
    dBrdz: float
    dBzdr: float
    dBzdz: float
    dBphidr: float
    dBphidz: float
    dns0dr: FloatArray
    dns0dz: FloatArray
    iexit: bool = False


@dataclass(frozen=True)
class PowerSettings:
    enabled: bool = True
    jeach: int = 0
    joutw: int = 1
    N: int = 3
    J: int = 8


@dataclass(frozen=True)
class ReferenceTrajectory:
    yyallray: FloatArray
    ist: int = 20
    ray_ids: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class RayTraceCase:
    equilibrium: Any
    frequency_hz: float
    initial_rays: FloatArray
    dt0: float
    nt0: int
    save_directory: Path | None = None
    power: PowerSettings = field(default_factory=PowerSettings)
    solver_rescale: float = 2.0
    label: str = ""
    reference: ReferenceTrajectory | None = None


@dataclass(frozen=True)
class RayTrajectory:
    legacy_matrix: FloatArray

    @property
    def states(self) -> FloatArray:
        return self.legacy_matrix[:, :6]

    @property
    def time(self) -> FloatArray:
        return self.legacy_matrix[:, 6]

    @property
    def magnetic_field(self) -> FloatArray:
        return self.legacy_matrix[:, 16]

    @property
    def psi(self) -> FloatArray:
        return self.legacy_matrix[:, -1]


@dataclass(frozen=True)
class PowerAbsorptionResult:
    sample_time: FloatArray
    sample_r: FloatArray
    sample_z: FloatArray
    cold_omega: ComplexArray
    hot_omega: ComplexArray
    absorbed_power: FloatArray
    species_count: int


@dataclass(frozen=True)
class SingleRayResult:
    initial_state: FloatArray
    traced_kr: float
    trajectory: RayTrajectory
    power: PowerAbsorptionResult | None = None


@dataclass(frozen=True)
class CaseResult:
    case: RayTraceCase
    rays: tuple[SingleRayResult, ...]
