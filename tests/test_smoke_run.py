from boray.examples import build_example_case
from boray.models import PowerSettings
from boray.tracing import run_case


def test_analytical_smoke_run():
    case = build_example_case(3, power=PowerSettings(enabled=False), dt0=0.002, nt0=10)
    result = run_case(case)
    assert len(result.rays) == 1
    assert result.rays[0].trajectory.legacy_matrix.shape[0] > 0
