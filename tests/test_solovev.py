from boray.solovev import SolovevParameters, create_solovev_equilibrium, sample_solovev_equilibrium


def test_solovev_equilibrium_builds():
    eq = create_solovev_equilibrium(SolovevParameters())
    point = sample_solovev_equilibrium(0.85, -0.2, eq)
    assert eq.fB.shape == eq.rr.shape
    assert eq.fns0.shape[0] == eq.S
    assert point.B > 0.0
    assert point.ns0.shape[0] == eq.S
