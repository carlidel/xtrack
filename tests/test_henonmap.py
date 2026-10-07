# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts

from _chaos_test_utils import context_and_line, sorted_coords
from _henon_reference import HenonReference

def test_henonmap_coefficients_and_validation():
    henon = xt.Henonmap(multipole_coeffs=[2.0, 6.0 * 0.3])
    # Kick of order 2 and 3: x^2 - y^2 + mu (x^3 - 3 x y^2) on px,
    # -2 x y + mu (y^3 - 3 x^2 y) on py
    fx = {(int(a), int(b)): c for a, b, c in zip(
        henon.fx_x_exps, henon.fx_y_exps, henon.fx_coeffs)}
    fy = {(int(a), int(b)): c for a, b, c in zip(
        henon.fy_x_exps, henon.fy_y_exps, henon.fy_coeffs)}
    assert fx == pytest.approx({(2, 0): 1., (0, 2): -1., (3, 0): .3, (1, 2): -.9})
    assert fy == pytest.approx({(1, 1): -2., (2, 1): -.9, (0, 3): .3})

    with pytest.raises(ValueError):
        xt.Henonmap(multipole_coeffs=np.ones(30))  # > 128 monomials
    with pytest.raises(ValueError):
        xt.Henonmap(twiss_params=[0, 1, 0])
    with pytest.raises(ValueError):
        xt.Henonmap(twiss_params=[0, -1, 0, 1])

    henon2 = xt.Henonmap.from_dict(henon.to_dict())
    for nn in ('fx_coeffs', 'fy_coeffs', 'twiss_params'):
        xo.assert_allclose(getattr(henon2, nn), getattr(henon, nn), rtol=0, atol=0)
    assert henon2.n_fx_coeffs == henon.n_fx_coeffs
    assert henon.allow_rot_and_shift is False
    assert henon.has_backtrack is True


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap(test_context):
    # Original test of the Henon map element (D. Veres, 2023), updated
    alpha_x = 1.0
    beta_x = 100.0
    alpha_y = 2.0
    beta_y = 10.0
    K2 = 0.1
    lmbd = K2 * beta_x**(3.0/2.0) / 2.0
    K3 = -5.0 * 3.0 * K2**2 * beta_x / 2.0

    N = 10
    x_n = np.linspace(0, 0.2, N)
    px_n = np.linspace(0, 0.2, N)
    x = x_n / lmbd * np.sqrt(beta_x)
    px = - alpha_x * x_n / np.sqrt(beta_x) / lmbd + px_n / np.sqrt(beta_x) / lmbd

    henon_n = xt.Henonmap(omega_x=2 * np.pi * 0.334,
                          omega_y=2 * np.pi * 0.279,
                          n_turns=1,
                          twiss_params=[0., 1., 0., 1.],
                          multipole_coeffs=[2.0, -30.0],
                          norm=True)
    ctx, line_n = context_and_line(test_context, [henon_n])
    henon = xt.Henonmap(omega_x=2 * np.pi * 0.334,
                        omega_y=2 * np.pi * 0.279,
                        n_turns=1,
                        twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
                        multipole_coeffs=[K2, K3],
                        norm=False)
    _, line = context_and_line(test_context, [henon])

    p_n = xt.Particles(x=x_n, px=px_n, _context=ctx)
    p = xt.Particles(x=x, px=px, _context=ctx)

    nTurns = 100
    x_n_all = np.zeros(N * nTurns)
    x_all = np.zeros(N * nTurns)
    px_n_all = np.zeros(N * nTurns)
    px_all = np.zeros(N * nTurns)
    for n in range(nTurns):
        line_n.track(p_n)
        line.track(p)
        x_n_all[n*N:(n+1)*N] = ctx.nparray_from_context_array(p_n.x)
        x_all[n*N:(n+1)*N] = ctx.nparray_from_context_array(p.x)
        px_n_all[n*N:(n+1)*N] = ctx.nparray_from_context_array(p_n.px)
        px_all[n*N:(n+1)*N] = ctx.nparray_from_context_array(p.px)
    x_all_n = x_all * lmbd / np.sqrt(beta_x)
    px_all_n = alpha_x * x_all / np.sqrt(beta_x) * lmbd + px_all * np.sqrt(beta_x) * lmbd

    xo.assert_allclose(x_n_all, x_all_n, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(px_n_all, px_all_n, atol=1e-15, rtol=1e-10)

    x_in_test  = np.asarray([1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 3, 3, 3, 0, 4]) * 0.01
    px_in_test = np.asarray([0, 1, 0, 0, 2, 0, 0, 2, 2, 0, 3, 3, 0, 3, 4]) * 0.01
    y_in_test  = np.asarray([0, 0, 1, 0, 0, 2, 0, 2, 0, 2, 3, 0, 3, 3, 4]) * 0.01
    py_in_test = np.asarray([0, 0, 0, 1, 0, 0, 2, 0, 2, 2, 0, 3, 3, 3, 4]) * 0.01

    omega_x = 2 * np.pi / 3.0
    omega_y = 2 * np.pi / 8.0

    sin_omega_x = np.sin(omega_x)
    cos_omega_x = np.cos(omega_x)
    sin_omega_y = np.sin(omega_y)
    cos_omega_y = np.cos(omega_y)

    x_out_test = cos_omega_x * x_in_test + sin_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
    px_out_test = -sin_omega_x * x_in_test + cos_omega_x * (px_in_test + (x_in_test**2 - y_in_test**2))
    y_out_test = cos_omega_y * y_in_test + sin_omega_y * (py_in_test - 2 * x_in_test * y_in_test)
    py_out_test = -sin_omega_y * y_in_test + cos_omega_y * (py_in_test - 2 * x_in_test * y_in_test)

    henon_test = xt.Henonmap(omega_x=omega_x,
                             omega_y=omega_y,
                             n_turns=1,
                             twiss_params=[0.0, 1.0, 0.0, 1.0],
                             multipole_coeffs=[2.0],
                             norm=True)
    _, line_test = context_and_line(test_context, [henon_test])

    p_test = xt.Particles(x=x_in_test, px=px_in_test, y=y_in_test,
                          py=py_in_test, _context=ctx)
    line_test.track(p_test)
    x_out, px_out, y_out, py_out = sorted_coords(ctx, p_test)

    xo.assert_allclose(x_out, x_out_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(px_out, px_out_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(y_out, y_out_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(py_out, py_out_test, atol=1e-15, rtol=1e-10)

    p_inv_test = xt.Particles(x=x_out, px=px_out, y=y_out, py=py_out, _context=ctx)
    line_test.track(p_inv_test, backtrack=True)
    x_in_inv, px_in_inv, y_in_inv, py_in_inv = sorted_coords(ctx, p_inv_test)

    xo.assert_allclose(x_in_inv, x_in_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(px_in_inv, px_in_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(y_in_inv, y_in_test, atol=1e-15, rtol=1e-10)
    xo.assert_allclose(py_in_inv, py_in_test, atol=1e-15, rtol=1e-10)


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap_one_turn_octupole(test_context):
    # Pilot convention: multipole_coeffs=[2, 6 mu]
    mu = -0.2
    omega_x, omega_y = 2 * np.pi * 0.168, 2 * np.pi * 0.201
    rng = np.random.default_rng(1)
    x, px, y, py = rng.uniform(-0.3, 0.3, size=(4, 50))

    henon = xt.Henonmap(omega_x=omega_x, omega_y=omega_y,
                        multipole_coeffs=[2.0, 6.0 * mu], norm=True)
    ctx, line = context_and_line(test_context, [henon])
    p = xt.Particles(x=x, px=px, y=y, py=py, _context=ctx)
    line.track(p)

    pxk = px + x**2 - y**2 + mu * (x**3 - 3 * x * y**2)
    pyk = py - 2 * x * y + mu * (y**3 - 3 * x**2 * y)
    expected = np.array([
        np.cos(omega_x) * x + np.sin(omega_x) * pxk,
        -np.sin(omega_x) * x + np.cos(omega_x) * pxk,
        np.cos(omega_y) * y + np.sin(omega_y) * pyk,
        -np.sin(omega_y) * y + np.cos(omega_y) * pyk,
    ])
    xo.assert_allclose(sorted_coords(ctx, p), expected, rtol=0, atol=1e-15)

    # The NumPy oracle implements the same map
    ref = HenonReference(omega_x, omega_y, [2.0, 6.0 * mu])
    xo.assert_allclose(ref.step(np.array([x, px, y, py])), expected,
                       rtol=0, atol=1e-15)


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap_backtrack_round_trip(test_context):
    # Physical coordinates, off-momentum, with chromaticity and dispersion
    henon = xt.Henonmap(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201,
                        twiss_params=[0.5, 20., -0.3, 5.],
                        dqx=2., dqy=-1., dx=1.5, ddx=0.1,
                        multipole_coeffs=[0.05, -0.01], norm=False)
    ctx, line = context_and_line(test_context, [henon])

    rng = np.random.default_rng(2)
    n_part = 20
    coords = dict(x=rng.uniform(-1e-2, 1e-2, n_part),
                  px=rng.uniform(-1e-3, 1e-3, n_part),
                  y=rng.uniform(-1e-2, 1e-2, n_part),
                  py=rng.uniform(-1e-3, 1e-3, n_part),
                  delta=rng.uniform(-1e-3, 1e-3, n_part))
    p = xt.Particles(_context=ctx, **coords)
    p0 = p.copy()

    line.track(p, num_turns=100)
    moved = sorted_coords(ctx, p)
    assert np.max(np.abs(moved - sorted_coords(ctx, p0))) > 1e-3
    line.track(p, num_turns=100, backtrack=True)

    xo.assert_allclose(sorted_coords(ctx, p), sorted_coords(ctx, p0),
                       rtol=0, atol=1e-12)
    xo.assert_allclose(sorted_coords(ctx, p, ('delta',)),
                       sorted_coords(ctx, p0, ('delta',)), rtol=0, atol=0)


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap_symplectic(test_context):
    # Finite-difference Jacobian of three turns in physical coordinates
    henon = xt.Henonmap(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201,
                        n_turns=3, twiss_params=[0.5, 20., -0.3, 5.],
                        multipole_coeffs=[0.05, -0.01], norm=False)
    ctx, line = context_and_line(test_context, [henon])

    z0 = np.array([[3e-3, 1e-4, -2e-3, 2e-4],
                   [-8e-3, -3e-4, 5e-3, 1e-4]])
    h = 1e-7
    names = ('x', 'px', 'y', 'py')
    for zz in z0:
        # 8 displaced copies: +h and -h along each coordinate
        init = np.tile(zz[:, None], (1, 8))
        for jj in range(4):
            init[jj, 2 * jj] += h
            init[jj, 2 * jj + 1] -= h
        p = xt.Particles(_context=ctx, **dict(zip(names, init)))
        line.track(p)
        out = sorted_coords(ctx, p)
        J = np.array([(out[:, 2 * jj] - out[:, 2 * jj + 1]) / (2 * h)
                      for jj in range(4)]).T
        S = np.array([[0, 1, 0, 0], [-1, 0, 0, 0],
                      [0, 0, 0, 1], [0, 0, -1, 0]])
        xo.assert_allclose(J.T @ S @ J, S, rtol=0, atol=1e-7)


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap_norm_equivalence(test_context):
    alpha_x, beta_x, alpha_y, beta_y = 0.7, 30., -1.2, 8.
    coeffs = [0.05, -0.01]
    omega = dict(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201)
    henon_phys = xt.Henonmap(**omega, twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
                             multipole_coeffs=coeffs, norm=False)
    # Same element, but expecting normalised coordinates at entry and exit
    henon_norm = xt.Henonmap(**omega, twiss_params=[alpha_x, beta_x, alpha_y, beta_y],
                             multipole_coeffs=coeffs, norm=True)
    ctx, line_phys = context_and_line(test_context, [henon_phys])
    _, line_norm = context_and_line(test_context, [henon_norm])

    rng = np.random.default_rng(3)
    x, y = rng.uniform(-1e-2, 1e-2, size=(2, 30))
    px, py = rng.uniform(-1e-3, 1e-3, size=(2, 30))
    sbx, sby = np.sqrt(beta_x), np.sqrt(beta_y)
    xn, pxn = x / sbx, alpha_x * x / sbx + px * sbx
    yn, pyn = y / sby, alpha_y * y / sby + py * sby

    p_phys = xt.Particles(x=x, px=px, y=y, py=py, _context=ctx)
    p_norm = xt.Particles(x=xn, px=pxn, y=yn, py=pyn, _context=ctx)
    line_phys.track(p_phys, num_turns=50)
    line_norm.track(p_norm, num_turns=50)

    xp_, pxp, yp, pyp = sorted_coords(ctx, p_phys)
    xo.assert_allclose(sorted_coords(ctx, p_norm), np.array([
        xp_ / sbx, alpha_x * xp_ / sbx + pxp * sbx,
        yp / sby, alpha_y * yp / sby + pyp * sby]), rtol=0, atol=1e-13)


@for_all_test_contexts
@allow_kernel_compilation
def test_henonmap_vs_numpy_reference(test_context):
    omega_x, omega_y = 2 * np.pi * 0.168, 2 * np.pi * 0.201
    coeffs = [2.0, 6.0 * (-0.2)]
    henon = xt.Henonmap(omega_x=omega_x, omega_y=omega_y,
                        multipole_coeffs=coeffs, norm=True)
    ctx, line = context_and_line(test_context, [henon])

    # Regular orbits close to the origin
    r = np.linspace(0.02, 0.2, 10)
    z0 = np.array([r, 0 * r, 0.5 * r, 0 * r])
    p = xt.Particles(_context=ctx, **dict(zip(('x', 'px', 'y', 'py'), z0)))
    line.track(p, num_turns=1000)

    ref = HenonReference(omega_x, omega_y, coeffs)
    z_ref = ref.track(z0, 1000)
    xo.assert_allclose(sorted_coords(ctx, p), z_ref, rtol=0, atol=1e-11)
