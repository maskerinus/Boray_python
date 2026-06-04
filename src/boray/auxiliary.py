"""Ports of the small standalone MATLAB helper scripts in `others/`."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from .constants import C2, EPSILON0, KB, ME, MP, QE, TWOPI


def coldwave_n2_vs_frequency(theta_deg: float = 70.0, B0: float = 0.9, n_e: float = 8e17):
    """Port of `coldwave_N2w.m`."""
    theta = math.radians(theta_deg)
    m_i = MP
    m_e = ME
    q_i = QE
    q_e = -QE
    n_i = abs(n_e * q_e / q_i)
    ww = 10.0 ** np.arange(5.0, 11.001, 0.001) * 1e1 * TWOPI

    wci = q_i * B0 / m_i
    wce = abs(q_e * B0 / m_e)
    wpi = math.sqrt(n_i * q_i**2 / (EPSILON0 * m_i))
    wpe = math.sqrt(n_e * q_e**2 / (EPSILON0 * m_e))
    wp = math.sqrt(wpi**2 + wpe**2)

    cos2theta = math.cos(theta) ** 2
    wR = wce / 2.0 + math.sqrt(wpe**2 + wce**2 / 4.0)
    wL = -wce / 2.0 + math.sqrt(wpe**2 + wce**2 / 4.0)
    nA2 = 1.0 + wpi**2 / wci**2 + wpe**2 / wce**2

    dielS = 1.0 - wpe**2 / (ww**2 - wce**2) - wpi**2 / (ww**2 - wci**2)
    dielD = -wce * wpe**2 / (ww * (ww**2 - wce**2)) + wci * wpi**2 / (ww * (ww**2 - wci**2))
    dielP = 1.0 - (wpe**2 + wpi**2) / (ww**2)
    dielR = dielS + dielD
    dielL = dielS - dielD
    dielA = dielS * math.sin(theta) ** 2 + dielP * cos2theta
    dielB = dielR * dielL * math.sin(theta) ** 2 + dielP * dielS * (1.0 + cos2theta)

    nn2 = np.zeros((2, ww.size), dtype=float)
    for jw in range(ww.size):
        roots = np.roots([dielA[jw], -dielB[jw], dielR[jw] * dielL[jw] * dielP[jw]])
        nn2[:, jw] = np.sort(np.real(roots))[::-1]

    return {
        "frequency_hz": ww / TWOPI,
        "n2": nn2,
        "fci": wci / TWOPI,
        "fce": wce / TWOPI,
        "fpi": wpi / TWOPI,
        "fpe": wpe / TWOPI,
        "fp": wp / TWOPI,
        "flh": 1.0 / math.sqrt(1.0 / ((wci / TWOPI) ** 2 + (wpi / TWOPI) ** 2) + 1.0 / ((wce / TWOPI) * (wci / TWOPI))),
        "fuh": math.sqrt((wce / TWOPI) ** 2 + (wpe / TWOPI) ** 2),
        "fR": wR / TWOPI,
        "fL": wL / TWOPI,
        "nA2": nA2,
    }


def coldwave_w_vs_k(theta_deg: float = 30.0, B0: float = 0.5, n_e: float = 1e19):
    """Port of `coldwave_wk.m`."""
    theta = math.radians(theta_deg)
    q_i = QE
    q_e = -QE
    m_i = MP
    m_e = ME
    n_i = abs(n_e * q_e / q_i)
    kc = 10.0 ** np.arange(-3.0, 3.001, 0.1) * 1e10

    wci = q_i * B0 / m_i
    wce = abs(q_e * B0 / m_e)
    wpi = math.sqrt(n_i * q_i**2 / (EPSILON0 * m_i))
    wpe = math.sqrt(n_e * q_e**2 / (EPSILON0 * m_e))
    wp = math.sqrt(wpi**2 + wpe**2)

    kc2 = kc**2
    kc4 = kc2**2
    wci2 = wci**2
    wce2 = wce**2
    wp2 = wp**2
    wextra = wp2 + wci * wce
    wextra2 = wextra**2
    cos2theta = math.cos(theta) ** 2

    polyc8 = -(2.0 * kc2 + wce2 + wci2 + 3.0 * wp2)
    polyc6 = kc4 + (2.0 * kc2 + wp2) * (wce2 + wci2 + 2.0 * wp2) + wextra2
    polyc4 = -(
        kc4 * (wce2 + wci2 + wp2)
        + 2.0 * kc2 * wextra2
        + kc2 * wp2 * (wce2 + wci2 - wci * wce) * (1.0 + cos2theta)
        + wp2 * wextra2
    )
    polyc2 = kc4 * (wp2 * (wce2 + wci2 - wci * wce) * cos2theta + wci * wce * wextra) + kc2 * wp2 * wci * wce * wextra * (1.0 + cos2theta)
    polyc0 = -kc4 * wci2 * wce2 * wp2 * cos2theta

    ww = np.zeros((10, kc.size), dtype=float)
    for jk in range(kc.size):
        roots = np.roots([1.0, 0.0, polyc8[jk], 0.0, polyc6[jk], 0.0, polyc4[jk], 0.0, polyc2[jk], 0.0, polyc0[jk]])
        ww[:, jk] = np.sort(np.real(roots))[::-1]

    return {"kc": kc, "omega": ww}


def cma_diagram(Mi: float = 1836.0, fw: float = 28e9, Bmin: float = 0.2, Bmax: float = 1.2, nemin: float = 5e15, nemax: float = 8e18):
    """Port of `plt_cma.m`."""
    x = 10.0 ** np.arange(-3.0, 8.00001, 0.00001)
    y = 10.0 ** np.arange(-2.0, 4.00001, 0.00001)

    xR = 1.0 / (1.0 / (1.0 - y) + 1.0 / (Mi + y))
    xL = 1.0 / (1.0 / (1.0 + y) + 1.0 / (Mi - y))
    xP = np.full_like(y, 1.0 / (1.0 + 1.0 / Mi))
    xS = 1.0 / (1.0 / (1.0 - y**2) + Mi / (Mi**2 - y**2))
    xRL_PS = (
        (1.0 / (1.0 - y) + 1.0 / (Mi + y))
        + (1.0 / (1.0 + y) + 1.0 / (Mi - y))
        - (1.0 + 1.0 / Mi)
        - (1.0 / (1.0 - y**2) + Mi / (Mi**2 - y**2))
    ) / (
        (1.0 / (1.0 - y) + 1.0 / (Mi + y)) * (1.0 / (1.0 + y) + 1.0 / (Mi - y))
        - (1.0 + 1.0 / Mi) * (1.0 / (1.0 - y**2) + Mi / (Mi**2 - y**2))
    )

    xR[xR < 0] = np.nan
    xL[xL < 0] = np.nan
    xS[xS < 0] = np.nan
    xRL_PS[xRL_PS < 0] = np.nan

    w = fw * TWOPI
    xmin = (nemin * QE**2 / (EPSILON0 * ME)) / w**2
    xmax = (nemax * QE**2 / (EPSILON0 * ME)) / w**2
    ymin = abs(-QE * Bmin / ME) / w
    ymax = abs(-QE * Bmax / ME) / w

    return {
        "x": x,
        "y": y,
        "xR": xR,
        "xL": xL,
        "xP": xP,
        "xS": xS,
        "xRL_PS": xRL_PS,
        "parameter_box": (xmin, xmax, ymin, ymax),
    }


def plot_cma_diagram(**kwargs):
    data = cma_diagram(**kwargs)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.loglog(data["xR"], data["y"], label="R=0")
    ax.loglog(data["xL"], data["y"], label="L=0")
    ax.loglog(data["xS"], data["y"], label="S=0")
    ax.loglog(data["xRL_PS"], data["y"], label="RL=PS")
    ax.loglog(data["xP"], data["y"], "--", label="P=0")
    xmin, xmax, ymin, ymax = data["parameter_box"]
    ax.plot([xmin, xmax], [ymin, ymax], ":", linewidth=2)
    ax.fill([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax], color="gold", alpha=0.08)
    ax.set_xlabel(r"x=$\omega_{pe}^2/\omega^2$")
    ax.set_ylabel(r"y=$|\omega_{ce}|/\omega$")
    ax.legend(frameon=False)
    return fig
