import numpy as np

from boray.equilibrium import cinterp2d, prepare_numerical_equilibrium, sample_numerical_equilibrium
from boray.models import NumericalEquilibrium


def test_cinterp2d_reconstructs_bilinear_surface():
    rg = np.array([0.0, 1.0, 2.0])
    zg = np.array([0.0, 1.0, 2.0])
    rr, zz = np.meshgrid(rg, zg, indexing="ij")
    frz = 2.0 + 3.0 * rr - 4.0 * zz + 5.0 * rr * zz
    fc = cinterp2d(rg, zg, frz)

    jr, jz = 0, 0
    hr, hz = 0.25, 0.75
    interp = fc[0, jr, jz] + fc[1, jr, jz] * hr + fc[2, jr, jz] * hz + fc[3, jr, jz] * hz * hr
    truth = 2.0 + 3.0 * (rg[jr] + hr) - 4.0 * (zg[jz] + hz) + 5.0 * (rg[jr] + hr) * (zg[jz] + hz)
    assert np.isclose(interp, truth)


def test_sample_numerical_equilibrium_matches_grid_value():
    rg = np.array([0.0, 1.0, 2.0])
    zg = np.array([0.0, 1.0, 2.0])
    rr, zz = np.meshgrid(rg, zg, indexing="ij")
    field = rr + 2.0 * zz
    zeros = np.zeros_like(field)

    eq = NumericalEquilibrium(
        rg=rg,
        zg=zg,
        dr=1.0,
        dz=1.0,
        rr=rr,
        zz=zz,
        fB=field,
        fBr=field,
        fBz=field,
        fBphi=zeros,
        fns0=np.stack([field], axis=0),
        fts0=np.stack([field], axis=0),
        fdBdr=np.ones_like(field),
        fdBdz=2.0 * np.ones_like(field),
        fdBrdr=np.ones_like(field),
        fdBrdz=2.0 * np.ones_like(field),
        fdBzdr=np.ones_like(field),
        fdBzdz=2.0 * np.ones_like(field),
        fdBphidr=zeros,
        fdBphidz=zeros,
        fdns0dr=np.stack([np.ones_like(field)], axis=0),
        fdns0dz=np.stack([2.0 * np.ones_like(field)], axis=0),
        fpsi=field,
        qs=np.array([1.0]),
        ms=np.array([1.0]),
        S=1,
    )

    eq = prepare_numerical_equilibrium(eq)
    point = sample_numerical_equilibrium(0.5, 0.25, eq)
    assert np.isclose(point.B, 1.0)
    assert np.isclose(point.ns0[0], 1.0)
