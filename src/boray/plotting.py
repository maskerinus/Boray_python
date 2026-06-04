"""Plot helpers that mirror the MATLAB BORAY figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import CaseResult


def _maybe_reference(case_result: CaseResult, ray_index: int):
    ref = case_result.case.reference
    if ref is None:
        return None
    ray_ids = tuple(ref.ray_ids)
    if ray_index >= len(ray_ids):
        return None
    ray_id = ray_ids[ray_index] - 1
    yyallray = np.asarray(ref.yyallray)
    if yyallray.ndim != 3:
        return None
    return yyallray[ref.ist :, :, ray_id]


def _save(fig: plt.Figure, output: str | Path | None) -> None:
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")


def plot_ray_result(case_result: CaseResult, ray_index: int = 0, output: str | Path | None = None) -> plt.Figure:
    """Create the main ray-tracing figure for one ray."""
    ray = case_result.rays[ray_index]
    eq = case_result.case.equilibrium
    legacy = ray.trajectory.legacy_matrix
    reference = _maybe_reference(case_result, ray_index)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    ax = axes.ravel()

    ax[0].contour(eq.rr, eq.zz, eq.fpsi, levels=40)
    ax[0].plot(legacy[:, 0], legacy[:, 2], linewidth=2)
    ax[0].plot(legacy[0, 0], legacy[0, 2], "rx", markersize=8, markeredgewidth=2)
    if reference is not None:
        ax[0].plot(reference[:, 0], reference[:, 2], "r:", linewidth=1.5)
    ax[0].set_xlabel("R")
    ax[0].set_ylabel("Z")
    ax[0].set_title("Ray In (R, Z)")

    cs = ax[1].contour(eq.rr, eq.zz, np.squeeze(eq.fns0[0]), levels=40)
    fig.colorbar(cs, ax=ax[1])
    ax[1].set_xlabel("R")
    ax[1].set_ylabel("Z")
    ax[1].set_title("n_s")

    ax[2].plot(legacy[:, 0] * np.cos(legacy[:, 1]), legacy[:, 0] * np.sin(legacy[:, 1]), linewidth=2)
    ax[2].plot(legacy[0, 0] * np.cos(legacy[0, 1]), legacy[0, 0] * np.sin(legacy[0, 1]), "rx", markersize=8, markeredgewidth=2)
    if reference is not None:
        ax[2].plot(reference[:, 0] * np.cos(reference[:, 1]), reference[:, 0] * np.sin(reference[:, 1]), "r:", linewidth=1.5)
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    ax[2].plot(np.min(eq.rg) * np.cos(theta), np.min(eq.rg) * np.sin(theta), color="0.4")
    ax[2].plot(np.max(eq.rg) * np.cos(theta), np.max(eq.rg) * np.sin(theta), color="0.4")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")
    ax[2].set_aspect("equal", adjustable="box")
    ax[2].set_title("Toroidal Projection")

    ax[3].plot(legacy[:, 0], legacy[:, 7], linewidth=2)
    ax[3].set_xlabel("R")
    ax[3].set_ylabel("D(w, k)")
    ax[3].set_title("Cold DR Residual")

    ax[4].plot(legacy[:, 6], legacy[:, 3], label="k_r", linewidth=2)
    ax[4].plot(legacy[:, 6], legacy[:, 4], label="n_phi", linewidth=2)
    ax[4].plot(legacy[:, 6], legacy[:, 5], label="k_z", linewidth=2)
    ax[4].set_xlabel("t")
    ax[4].set_ylabel("k")
    ax[4].legend(frameon=False)
    ax[4].set_title(f"f = {case_result.case.frequency_hz / 1e6:.3f} MHz")

    if reference is not None:
        xref = np.arange(reference.shape[0])
        ax[5].plot(xref, reference[:, 3], label="k_r", linewidth=2)
        ax[5].plot(xref, reference[:, 4], label="n_phi", linewidth=2)
        ax[5].plot(xref, reference[:, 5], label="k_z", linewidth=2)
        ax[5].set_title("GENRAY Reference")
    else:
        ax[5].plot(legacy[:, 6], legacy[:, 8], label="vg_r", linewidth=2)
        ax[5].plot(legacy[:, 6], legacy[:, 9] * legacy[:, 0], label="vg_phi", linewidth=2)
        ax[5].plot(legacy[:, 6], legacy[:, 10], label="vg_z", linewidth=2)
        ax[5].set_title("Group Velocity")
    ax[5].set_xlabel("t")
    ax[5].set_ylabel("value")
    ax[5].legend(frameon=False)

    _save(fig, output)
    return fig


def plot_power_result(case_result: CaseResult, ray_index: int = 0, output: str | Path | None = None) -> plt.Figure:
    """Create the hot-plasma power plot for one ray."""
    ray = case_result.rays[ray_index]
    if ray.power is None:
        raise ValueError("This case was run without power absorption enabled.")

    eq = case_result.case.equilibrium
    legacy = ray.trajectory.legacy_matrix
    power = ray.power
    reference = _maybe_reference(case_result, ray_index)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    ax = axes.ravel()

    ax[0].plot(power.sample_time, np.real(power.cold_omega), linewidth=2, label="cold DR")
    ax[0].plot(power.sample_time, np.real(power.hot_omega[:, 0]), linewidth=2, label="hot DR")
    for js in range(1, power.species_count):
        ax[0].plot(power.sample_time, np.real(power.hot_omega[:, js]), "--", linewidth=1.5)
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("omega_r")
    ax[0].legend(frameon=False)

    ax[1].plot(power.sample_time, np.imag(power.hot_omega[:, 0]), linewidth=2, label="all")
    if power.species_count > 1:
        for js in range(1, power.species_count):
            ax[1].plot(power.sample_time, np.imag(power.hot_omega[:, js]), "--", linewidth=1.5, label=f"s={js}")
        ax[1].plot(power.sample_time, np.sum(np.imag(power.hot_omega[:, 1:]), axis=1), ":", linewidth=2, label="sum")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("omega_i")
    ax[1].legend(frameon=False)

    ax[2].plot(power.sample_r, 1.0 - power.absorbed_power[:, 0], linewidth=2, label="BORAY")
    if reference is not None and reference.shape[1] > 20:
        ax[2].plot(reference[:, 0], reference[:, -1] / reference[0, -1], "--", linewidth=1.5, label="GENRAY")
    ax[2].set_xlabel("R")
    ax[2].set_ylabel("Power Absorb")
    ax[2].legend(frameon=False)

    ax[3].contour(eq.rr, eq.zz, eq.fpsi, levels=40)
    ax[3].plot(legacy[:, 0], legacy[:, 2], linewidth=2)
    ax[3].plot(legacy[0, 0], legacy[0, 2], "rx", markersize=8, markeredgewidth=2)
    ax[3].set_xlabel("R")
    ax[3].set_ylabel("Z")
    ax[3].set_title("Psi And Ray")

    cs_b = ax[4].contour(eq.rr, eq.zz, eq.fB, levels=40)
    fig.colorbar(cs_b, ax=ax[4])
    ax[4].set_xlabel("R")
    ax[4].set_ylabel("Z")
    ax[4].set_title("B")

    cs_n = ax[5].contour(eq.rr, eq.zz, np.squeeze(eq.fns0[0]), levels=40)
    fig.colorbar(cs_n, ax=ax[5])
    ax[5].plot(power.sample_r, power.sample_z, linewidth=2)
    if reference is not None:
        ax[5].plot(reference[:, 0], reference[:, 2], "r--", linewidth=1.5)
    ax[5].set_xlabel("R")
    ax[5].set_ylabel("Z")
    ax[5].set_title("n_e")

    _save(fig, output)
    return fig
