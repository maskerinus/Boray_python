"""Command-line entrypoint for the BORAY Python port."""

from __future__ import annotations

import argparse
from pathlib import Path

from .examples import build_example_case
from .models import PowerSettings
from .plotting import plot_power_result, plot_ray_result
from .tracing import run_case


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BORAY Python port.")
    parser.add_argument("--icase", type=int, default=2, choices=(1, 2, 3), help="Example case from the original MATLAB entrypoint.")
    parser.add_argument("--matlab-root", type=Path, default=None, help="Path to the cloned BORAY MATLAB repository.")
    parser.add_argument("--dt0", type=float, default=None, help="Override the base time step.")
    parser.add_argument("--nt0", type=int, default=None, help="Override the base step count.")
    parser.add_argument("--no-power", action="store_true", help="Disable hot-plasma absorption post-processing.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated plots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    power = PowerSettings(enabled=not args.no_power)
    case = build_example_case(args.icase, matlab_root=args.matlab_root, power=power, dt0=args.dt0, nt0=args.nt0)
    result = run_case(case)

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, _ray in enumerate(result.rays):
            ray_plot = output_dir / f"ray_{idx + 1}.png"
            plot_ray_result(result, ray_index=idx, output=ray_plot)
            if result.rays[idx].power is not None:
                power_plot = output_dir / f"power_{idx + 1}.png"
                plot_power_result(result, ray_index=idx, output=power_plot)

    print(f"Completed {case.label} with {len(result.rays)} ray(s).")
    for idx, ray in enumerate(result.rays, start=1):
        print(f"ray {idx}: traced_kr={ray.traced_kr:.6g}, steps={ray.trajectory.legacy_matrix.shape[0]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
