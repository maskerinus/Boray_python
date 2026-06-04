# BORAY Python

This directory contains a Python port of the MATLAB code from [`hsxie/boray`](https://github.com/hsxie/boray).

The port keeps the original structure but expresses it as a small Python package:

- `boray.equilibrium`
  Bilinear interpolation and numerical equilibrium sampling.
- `boray.solovev`
  Analytical Solovev equilibrium generation.
- `boray.dispersion`
  Cold and hot plasma dispersion solvers plus Hamiltonian derivatives.
- `boray.tracing`
  Initial `k_r` solve, RK4 ray tracing, and power-absorption post-processing.
- `boray.plotting`
  Ray and power plots analogous to the MATLAB figures.
- `boray.io`
  MATLAB `.mat` equilibrium loading and auxiliary data readers.
