# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo

import _henon_reference as hr

OMEGA_X = 2 * np.pi * 0.168
OMEGA_Y = 2 * np.pi * 0.201
COEFFS = [2.0]  # Quadratic Henon map (mu = 0)

R_REGULAR = 0.1
R_CHAOTIC = 0.36


def _diagonal_orbits(r):
    r = np.atleast_1d(r).astype(float)
    return np.array([r, 0 * r, r, 0 * r])


# NumPy oracle --------------------------------------------------------------

def test_reference_jacobian():
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, [2.0, 6.0 * (-0.2)])
    z0 = np.array([[0.1, -0.2], [0.05, 0.1], [-0.15, 0.2], [0.02, 0.0]])
    J = ref.jacobian(z0)
    h = 1e-6
    for jj in range(4):
        dz = np.zeros((4, 1))
        dz[jj] = h
        fd = (ref.step(z0 + dz) - ref.step(z0 - dz)) / (2 * h)
        xo.assert_allclose(J[:, :, jj], fd.T, rtol=0, atol=1e-9)
    S = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    for Ji in J:
        xo.assert_allclose(Ji.T @ S @ Ji, S, rtol=0, atol=1e-14)
    xo.assert_allclose(ref.step_inverse(ref.step(z0)), z0, rtol=0, atol=1e-15)


def test_reference_linear_map():
    # Pure rotation: no stretching, angles preserved, tunes exact
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, [0.0])
    z0 = _diagonal_orbits([0.1, 0.3])
    res = hr.tangent_indicators(ref, z0, 200, [50, 200])
    xo.assert_allclose(res['log_growth'], 0, rtol=0, atol=1e-13)
    xo.assert_allclose(res['fli_birkhoff'], 0, rtol=0, atol=1e-13)
    xo.assert_allclose(res['sali'], np.sqrt(2), rtol=0, atol=1e-13)
    xo.assert_allclose(res['gali'], 1, rtol=0, atol=1e-13)

    w = hr.birkhoff_weights(100)
    xo.assert_allclose(w.sum(), 1, rtol=1e-15, atol=0)
    xo.assert_allclose(w, w[::-1], rtol=1e-12, atol=0)
    qx1, qy1, qx2, qy2, _ = hr.birkhoff_tunes(ref.trajectory(z0, 400), 0, 200)
    for qx in (qx1, qx2):
        xo.assert_allclose(qx, 0.168, rtol=0, atol=1e-13)
    for qy in (qy1, qy2):
        xo.assert_allclose(qy, 0.201, rtol=0, atol=1e-13)


def test_reference_renormalisation_schedule():
    # Renormalising less often changes neither the total stretching nor the
    # directions of the deviation vectors (only the Birkhoff FLI changes).
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS)
    z0 = _diagonal_orbits([R_REGULAR, R_CHAOTIC])
    res1 = hr.tangent_indicators(ref, z0, 300, [100, 300], renorm_every=1)
    res10 = hr.tangent_indicators(ref, z0, 300, [100, 300], renorm_every=10)
    xo.assert_allclose(res10['log_growth'], res1['log_growth'], rtol=1e-12, atol=1e-11)
    xo.assert_allclose(res10['sali'], res1['sali'], rtol=1e-8, atol=1e-14)
    xo.assert_allclose(res10['gali'], res1['gali'], rtol=1e-6, atol=1e-30)
    assert np.all(hr.renormalisation_turns(25, 10, [5, 25]) == [5, 10, 20, 25])


def test_reference_regular_vs_chaotic():
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS)
    z0 = _diagonal_orbits([R_REGULAR, R_CHAOTIC])
    res = hr.tangent_indicators(ref, z0, 1000, [1000])
    fli = res['fli'][0]
    assert fli[0] < 5 and fli[1] > 15
    assert res['sali'][0, 0] > 1e-2 and res['sali'][0, 1] < 1e-4
    assert res['gali'][0, 0, 2] > 1e-4 and res['gali'][0, 1, 2] < 1e-20
    rem = hr.reversibility_error(ref, z0, 1000)
    assert rem[0] < 1e-12 and rem[1] > 1e-6

    # Small amplitude: tunes close to the linear tunes
    traj = ref.trajectory(_diagonal_orbits(1e-3), 2000)
    qx1, qy1, qx2, qy2, diff = hr.birkhoff_tunes(traj, 0, 1000)
    xo.assert_allclose(qx1, 0.168, rtol=0, atol=1e-5)
    xo.assert_allclose(qy1, 0.201, rtol=0, atol=1e-5)
    assert diff[0] < -10
