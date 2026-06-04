"""Shared physical constants used by the BORAY port."""

from __future__ import annotations

import math

C = 2.99792458e8
C2 = C * C
EPSILON0 = 8.854187817e-12
MU0 = 1.0 / (C2 * EPSILON0)
KB = 1.38064852e-23
QE = 1.60217662e-19
MP = 1.6726219e-27
ME = 9.1094e-31
TWOPI = 2.0 * math.pi
EV_TO_J = QE
EV_TO_K = EV_TO_J / KB
