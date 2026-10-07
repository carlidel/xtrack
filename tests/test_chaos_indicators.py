# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts

import _henon_reference as hr
from _chaos_test_utils import context_and_line

OMEGA_X = 2 * np.pi * 0.168
OMEGA_Y = 2 * np.pi * 0.201
COEFFS = [2.0]  # Quadratic Henon map (mu = 0)

R_REGULAR = 0.1
R_CHAOTIC = 0.36


def _diagonal_orbits(r):
    r = np.atleast_1d(r).astype(float)
    return np.array([r, 0 * r, r, 0 * r])


def _henon_line(test_context, aperture=1.0):
    elements = [xt.Henonmap(omega_x=OMEGA_X, omega_y=OMEGA_Y,
                            multipole_coeffs=COEFFS, norm=True)]
    if aperture is not None:
        elements.append(xt.LimitRect(min_x=-aperture, max_x=aperture,
                                     min_y=-aperture, max_y=aperture))
    return context_and_line(test_context, elements)


def _particles(context, z0, particle_id=None):
    kwargs = dict(zip(('x', 'px', 'y', 'py'), z0))
    if particle_id is not None:
        kwargs['particle_id'] = particle_id
    return xt.Particles(_context=context, **kwargs)


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


# Ghosts and FLI ------------------------------------------------------------

def test_ghost_layout():
    p = xt.Particles(x=[0.3, 0.1, 0.2], y=[0.0, 0.1, 0.2],
                     particle_id=[7, 3, 5])
    metric = np.diag([1.0, 2.0, 4.0, 8.0])
    p_all, layout = xt.chaos.build_ghost_particles(
        p, n_ghosts=3, displacement=1e-6, metric=metric)
    assert layout.n_ref == 3 and layout.num_particles == 12
    assert np.all(layout.reference_particle_id == [3, 5, 7])
    assert np.all(p_all.particle_id == np.arange(12))
    assert np.all(p_all.parent_particle_id ==
                  [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert np.all(layout.ghost_id(1, np.arange(3)) == [6, 7, 8])
    xo.assert_allclose(p_all.x[:3], [0.1, 0.2, 0.3], rtol=0, atol=0)
    # Ghost g displaced by eps along metric^-1 e_g
    dz = np.array([getattr(p_all, nn)[3:] - getattr(p_all, nn)[
        p_all.parent_particle_id[3:]] for nn in ('x', 'px', 'y', 'py')])
    expected = np.tile(1e-6 * np.linalg.inv(metric)[:, :3], (1, 3))
    xo.assert_allclose(dz, expected, rtol=1e-8, atol=1e-22)
    # Input untouched
    assert np.all(p.particle_id == [7, 3, 5])

    with pytest.raises(ValueError):
        xt.chaos.build_ghost_particles(p, n_ghosts=5)
    p.state[1] = 0
    with pytest.raises(ValueError):
        xt.chaos.build_ghost_particles(p, n_ghosts=2)


def test_birkhoff_weights_and_schedule():
    for nn in (1, 7, 1000):
        xo.assert_allclose(xt.chaos.birkhoff_weights(nn),
                           hr.birkhoff_weights(nn), rtol=1e-14, atol=0)
    xo.assert_allclose(xt.chaos.renormalisation_turns(25, 10, [5, 25]),
                       hr.renormalisation_turns(25, 10, [5, 25]), rtol=0, atol=0)


@for_all_test_contexts
@allow_kernel_compilation
def test_fli_ghosts_vs_tangent_map(test_context):
    # Regular orbits: the ghost FLI converges to the exact-tangent FLI with
    # an error O(eps) for eps >= 1e-7 (round-off ~1e-16 |z| / eps dominates
    # below 1e-8; see the report for the full scan).
    ctx, line = _henon_line(test_context)
    r = np.linspace(0.05, 0.25, 5)
    z0 = _diagonal_orbits(r)
    num_turns, sample_turns = 500, [100, 500]
    ref = hr.tangent_indicators(hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS),
                                z0, num_turns, sample_turns)

    err = {}
    for eps in (1e-5, 1e-6, 1e-8):
        res = xt.chaos.compute_tangent_indicators(
            line, _particles(ctx, z0), num_turns, sample_turns,
            n_ghosts=4, displacement=eps)
        assert np.all(res.valid) and np.all(res.lost_at_turn == -1)
        assert np.all(res.particle_id == np.arange(5))
        xo.assert_allclose(res.lyapunov, res.fli / np.array(sample_turns)[:, None],
                           rtol=1e-15, atol=0)
        err[eps] = np.max(np.abs(res.log_growth - ref['log_growth']))
        xo.assert_allclose(res.fli_birkhoff, ref['fli_birkhoff'],
                           rtol=0, atol=10 * eps + 1e-8)
    assert err[1e-8] < 2e-6
    assert err[1e-6] < 1e-4
    assert 5 < err[1e-5] / err[1e-6] < 20  # O(eps)

    # Chaotic orbit: same FLI up to the finite-difference error
    z1 = _diagonal_orbits(R_CHAOTIC)
    res = xt.chaos.compute_tangent_indicators(
        line, _particles(ctx, z1), 1000, [1000], n_ghosts=1)
    ref = hr.tangent_indicators(hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS),
                                z1, 1000, [1000], n_vectors=1)
    xo.assert_allclose(res.fli, ref['fli'], rtol=1e-4, atol=0)
    assert not hasattr(res, 'sali')


@for_all_test_contexts
@allow_kernel_compilation
def test_tangent_indicators_serial_vs_openmp(test_context):
    z0 = _diagonal_orbits(np.linspace(0.05, 0.4, 8))
    out = []
    for tc in (test_context, xo.ContextCpu()):
        ctx, line = _henon_line(tc)
        out.append(xt.chaos.compute_tangent_indicators(
            line, _particles(ctx, z0), 300, [50, 300], n_ghosts=4,
            renorm_every=3))
    for name in ('valid', 'lost_at_turn', 'log_growth', 'fli_birkhoff', 'sali'):
        xo.assert_allclose(getattr(out[0], name), getattr(out[1], name),
                           rtol=0, atol=0)
    for kk in (2, 3, 4):
        xo.assert_allclose(out[0].gali[kk], out[1].gali[kk], rtol=0, atol=0)


@for_all_test_contexts
@allow_kernel_compilation
def test_tangent_indicators_losses(test_context):
    aperture = 0.6
    ctx, line = _henon_line(test_context, aperture=aperture)

    # Grid straddling the stability border
    rng = np.random.default_rng(11)
    r = np.linspace(0.25, 0.55, 40)
    z0 = _diagonal_orbits(r)
    num_turns, sample_turns = 400, [50, 400]

    # Losses of the references alone
    p_ref = _particles(ctx, z0)
    line.track(p_ref, num_turns=num_turns)
    state = ctx.nparray_from_context_array(p_ref.state)
    order = np.argsort(ctx.nparray_from_context_array(p_ref.particle_id))
    ref_lost = state[order] <= 0
    ref_lost_turn = ctx.nparray_from_context_array(p_ref.at_turn)[order]
    assert 5 < ref_lost.sum() < 35

    res = xt.chaos.compute_tangent_indicators(
        line, _particles(ctx, z0), num_turns, sample_turns, n_ghosts=4)
    invalid = ~res.valid[-1]
    assert np.all(invalid[ref_lost])
    assert np.all(res.lost_at_turn[~invalid] == -1)
    assert np.all(res.lost_at_turn[invalid] >= 0)
    assert np.all(res.lost_at_turn[ref_lost] <= ref_lost_turn[ref_lost])
    # The ghosts are lost with their reference except near the border
    assert np.mean(res.lost_at_turn[ref_lost] == ref_lost_turn[ref_lost]) > 0.8
    # Orbits valid at a sample turn were not lost before it
    for ss, tt in enumerate(sample_turns):
        assert np.all(res.valid[ss] == ((res.lost_at_turn == -1)
                                         | (res.lost_at_turn >= tt)))
    # No NaN for survivors, NaN for orbits lost before the sample turn
    for name in ('fli', 'fli_birkhoff', 'sali'):
        vv = getattr(res, name)
        assert np.all(np.isfinite(vv[res.valid]))
        assert np.all(np.isnan(vv[~res.valid]))
    assert np.all(np.isfinite(res.gali[4][res.valid]))

    # Invariance under reordering of the input particles
    perm = rng.permutation(len(r))
    ids = np.arange(len(r))
    res_perm = xt.chaos.compute_tangent_indicators(
        line, _particles(ctx, z0[:, perm], particle_id=ids[perm]),
        num_turns, sample_turns, n_ghosts=4)
    assert np.all(res_perm.particle_id == ids)
    for name in ('valid', 'lost_at_turn', 'fli', 'fli_birkhoff', 'sali'):
        xo.assert_allclose(getattr(res_perm, name), getattr(res, name),
                           rtol=0, atol=0)


# SALI and GALI -------------------------------------------------------------

def _window_slope(n, values, edges):
    """Slope of log(median of values) vs log(n) over windows (a, b]."""
    n = np.asarray(n)
    centres, medians = [], []
    for aa, bb in zip(edges[:-1], edges[1:]):
        sel = (n > aa) & (n <= bb)
        centres.append(np.log(np.sqrt(aa * bb)))
        medians.append(np.median(np.log(values[sel])))
    return np.polyfit(centres, medians, 1)[0]


@for_all_test_contexts
@allow_kernel_compilation
def test_sali_gali_regular_orbit(test_context):
    # On a 2-torus of the 4D map: SALI and GALI_2 ~ constant,
    # GALI_3 ~ n^-2, GALI_4 ~ n^-4. The power laws set in after a transient
    # that grows as the amplitude-dependent detuning decreases; r = 0.25 is
    # in the asymptotic regime from ~1e3 turns. Directions do not depend on
    # the renormalisation period, so renorm_every=100 is exact here.
    ctx, line = _henon_line(test_context)
    z0 = _diagonal_orbits(0.25)
    num_turns = 20_000
    sample_turns = np.arange(100, num_turns + 1, 100)
    res = xt.chaos.compute_tangent_indicators(
        line, _particles(ctx, z0), num_turns, sample_turns, n_ghosts=4,
        renorm_every=100)
    ref = hr.tangent_indicators(hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS),
                                z0, num_turns, sample_turns, renorm_every=100)
    assert np.all(res.valid)
    # Measured over 2e4 turns: 5e-4 (SALI, GALI_2), 1e-2 (GALI_3, GALI_4,
    # values down to ~1e-9)
    xo.assert_allclose(res.sali, ref['sali'], rtol=2e-3, atol=0)
    xo.assert_allclose(res.gali[2], ref['gali'][:, :, 0], rtol=2e-3, atol=0)
    for kk in (3, 4):
        xo.assert_allclose(res.gali[kk], ref['gali'][:, :, kk - 2],
                           rtol=3e-2, atol=0)

    # SALI oscillates along the torus: fit the medians of octave windows.
    # Measured: +0.09 (SALI, GALI_2), -1.74 (GALI_3), -3.91 (GALI_4).
    edges = (1250, 2500, 5000, 10_000, 20_000)
    assert np.min(res.sali) > 1e-2
    assert abs(_window_slope(sample_turns, res.sali[:, 0], edges)) < 0.3
    assert abs(_window_slope(sample_turns, res.gali[2][:, 0], edges)) < 0.3
    assert -2.3 < _window_slope(sample_turns, res.gali[3][:, 0], edges) < -1.6
    assert -4.5 < _window_slope(sample_turns, res.gali[4][:, 0], edges) < -3.5


@for_all_test_contexts
@allow_kernel_compilation
def test_sali_gali_chaotic_orbit(test_context):
    # For a 4D symplectic map the Lyapunov spectrum is (l1, l2, -l2, -l1):
    # GALI_3 ~ exp(-2 l1 n) and GALI_4 ~ exp(-4 l1 n), with l1 the FLI slope.
    ctx, line = _henon_line(test_context)
    z0 = _diagonal_orbits(0.38)
    sample_turns = np.arange(5, 1001, 5)
    res = xt.chaos.compute_tangent_indicators(
        line, _particles(ctx, z0), 1000, sample_turns, n_ghosts=4)
    ref = hr.tangent_indicators(hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS),
                                z0, 1000, sample_turns)
    assert np.all(res.valid)

    fit = (sample_turns >= 10) & (sample_turns <= 110)
    lyap = np.polyfit(sample_turns[fit], res.fli[fit, 0], 1)[0]
    assert lyap > 0.05
    for kk, factor in ((3, 2), (4, 4)):
        slope = np.polyfit(sample_turns[fit], np.log(res.gali[kk][fit, 0]), 1)[0]
        assert 0.85 < slope / (-factor * lyap) < 1.15
    sali_slope = np.polyfit(sample_turns[fit], np.log(res.sali[fit, 0]), 1)[0]
    assert -1.2 * lyap < sali_slope < 0  # rate l1 - l2 <= l1
    assert res.sali[-1, 0] < 1e-5

    # Above the finite-difference floor the ghosts reproduce the tangent map
    above = ref['gali'][:, 0, 2] > 1e-12
    assert above.sum() >= 15
    xo.assert_allclose(res.gali[4][above, 0], ref['gali'][above, 0, 2],
                       rtol=0.1, atol=0)
    # Floor: with eps = 1e-8 the ghost directions are resolved to
    # ~1e-16 |z| / eps, so GALI_4 saturates (or drops to exactly 0 when two
    # ghosts coincide) while the exact-tangent value keeps decreasing.
    assert ref['gali'][-1, 0, 2] < 1e-35
    late = res.gali[4][sample_turns >= 500, 0]
    assert np.median(late[late > 0]) > 1e-26


# REM -----------------------------------------------------------------------

@for_all_test_contexts
@allow_kernel_compilation
def test_rem(test_context):
    r = np.array([0.05, 0.1, 0.2, 0.25, 0.34, 0.36, 0.38, 0.5])
    regular = r <= 0.25
    z0 = _diagonal_orbits(r)
    rem_turns = [10, 1000, 3000]

    res_all = []
    for tc in (test_context, xo.ContextCpu()):
        ctx, line = _henon_line(tc)
        res_all.append(xt.chaos.compute_rem(line, _particles(ctx, z0), rem_turns))
    res = res_all[0]
    assert np.all(res.rem_turns == rem_turns)
    assert np.all(res.particle_id == np.arange(len(r)))

    # Lost orbit (r = 0.5): NaN
    assert np.all(np.isnan(res.rem[:, -1])) and not np.any(res.rem_valid[:, -1])
    assert np.all(res.rem_valid[:, :-1])

    # Regular orbits: near round-off, slow growth
    assert np.all(res.rem[0, regular] < 1e-15)
    assert np.all(res.rem[1, regular] < 1e-12)
    # Chaotic orbits: exponential growth, saturating at O(1)
    chaotic = ~regular & (r < 0.5)
    assert np.all(res.rem[1, chaotic] > 1e-6)
    assert np.all(res.rem[2, chaotic] > 1e-1)

    # Same orders of magnitude as the NumPy oracle on regular orbits
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS)
    rem_ref = hr.reversibility_error(ref, z0[:, regular], 1000)
    assert np.all(np.abs(np.log10(res.rem[1, regular] / rem_ref)) < 1.5)

    # REM is round-off driven: compare contexts by classification
    for res_ctx in res_all:
        assert np.all((res_ctx.rem[1, :-1] < 1e-10) == regular[:-1])


@for_all_test_contexts
@allow_kernel_compilation
def test_rem_requires_backtrackable_line(test_context):
    line = xt.Line(elements=[
        xt.Henonmap(omega_x=OMEGA_X, omega_y=OMEGA_Y, multipole_coeffs=COEFFS,
                    norm=True),
        xt.LineSegmentMap(qx=0.31, qy=0.32)])
    line.build_tracker(_context=test_context, compile=False)
    p = xt.Particles(x=[0.1], y=[0.1], _context=test_context)
    with pytest.raises(ValueError, match='LineSegmentMap'):
        xt.chaos.compute_rem(line, p, 10)

    ctx, line_ok = _henon_line(test_context)
    line_ok.enable_time_dependent_vars = True
    with pytest.raises(ValueError, match='time-dependent'):
        xt.chaos.compute_rem(line_ok, p, 10)


# FMA -----------------------------------------------------------------------

@for_all_test_contexts
@allow_kernel_compilation
def test_fma_birkhoff_tune_monitor(test_context):
    r = np.array([1e-3, 0.05, 0.1, 0.2, 0.34, 0.38, 0.5])
    regular = r <= 0.2
    chaotic = (r > 0.3) & (r < 0.45)
    z0 = _diagonal_orbits(r)
    window = 500

    # In-line monitor at the start of the line: one tracking call
    monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=len(r), window=window)
    henon = xt.Henonmap(omega_x=OMEGA_X, omega_y=OMEGA_Y,
                        multipole_coeffs=COEFFS, norm=True)
    aperture = xt.LimitRect(min_x=-1, max_x=1, min_y=-1, max_y=1)
    ctx, line_mon = context_and_line(test_context, [monitor, henon, aperture])
    res = xt.chaos.compute_fma(line_mon, _particles(ctx, z0), window)

    # Standalone monitor between one-turn tracking calls: identical
    _, line = _henon_line(test_context)
    res_sa = xt.chaos.compute_fma(line, _particles(ctx, z0), window)
    for name in ('qx', 'qy', 'qx2', 'qy2', 'tune_diffusion'):
        xo.assert_allclose(getattr(res_sa, name), getattr(res, name),
                           rtol=0, atol=0)

    # Lost particle
    assert not res.fma_valid[-1] and np.isnan(res.qx[-1])
    assert np.all(res.fma_valid[:-1])

    # NumPy oracle on the regular orbits
    ref = hr.HenonReference(OMEGA_X, OMEGA_Y, COEFFS)
    qx1, qy1, qx2, qy2, diff = hr.birkhoff_tunes(
        ref.trajectory(z0[:, regular], 2 * window), 0, window)
    for got, exp in ((res.qx, qx1), (res.qy, qy1), (res.qx2, qx2), (res.qy2, qy2)):
        xo.assert_allclose(got[regular], exp, rtol=0, atol=1e-12)

    # Small amplitude: linear tunes
    xo.assert_allclose(res.qx[0], 0.168, rtol=0, atol=1e-6)
    xo.assert_allclose(res.qy[0], 0.201, rtol=0, atol=1e-6)

    # Diffusion separates regular from chaotic orbits
    assert np.all(res.tune_diffusion[regular] < -6)
    assert np.all(res.tune_diffusion[chaotic] > -4.5)

    # Birkhoff vs NAFF (reference path) on regular orbits
    res_naff = xt.chaos.compute_fma(line, _particles(ctx, z0), window,
                                    method='naff')
    assert res_naff.fma_method == 'naff'
    assert not res_naff.fma_valid[-1]
    for name in ('qx', 'qy', 'qx2', 'qy2'):
        xo.assert_allclose(getattr(res_naff, name)[regular],
                           getattr(res, name)[regular], rtol=0, atol=1e-5)
    assert np.all(res_naff.tune_diffusion[chaotic] > -4.5)

    # The monitor is passive in backtracking and the line stays backtrackable
    qx_before = monitor.get_tunes()[0].copy()
    p = _particles(ctx, z0[:, :3])
    line_mon.track(p, num_turns=5, backtrack=True)
    xo.assert_allclose(monitor.get_tunes()[0], qx_before, rtol=0, atol=0)

    # A monitor that does not match the request is rejected
    with pytest.raises(ValueError, match='window'):
        xt.chaos.compute_fma(line_mon, _particles(ctx, z0), window + 1)


# Combined driver -----------------------------------------------------------

@for_all_test_contexts
@allow_kernel_compilation
def test_compute_indicators(test_context):
    r = np.array([0.05, 0.2, 0.36, 0.5])
    z0 = _diagonal_orbits(r)
    num_turns, sample_turns, window = 401, [100, 401], 200

    monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=len(r), window=window)
    henon = xt.Henonmap(omega_x=OMEGA_X, omega_y=OMEGA_Y,
                        multipole_coeffs=COEFFS, norm=True)
    aperture = xt.LimitRect(min_x=-1, max_x=1, min_y=-1, max_y=1)
    ctx, line_mon = context_and_line(test_context, [monitor, henon, aperture])
    _, line = _henon_line(test_context)

    for ll, fma_where in ((line_mon, 'birkhoff (in-line, ghost pass)'),
                          (line, 'birkhoff')):
        res = xt.chaos.compute_indicators(
            ll, _particles(ctx, z0), num_turns, sample_turns,
            fma_window=window)
        assert res.fma_method == fma_where
        tan = xt.chaos.compute_tangent_indicators(
            line, _particles(ctx, z0), num_turns, sample_turns)
        for name in ('valid', 'lost_at_turn', 'fli', 'lyapunov',
                     'fli_birkhoff', 'sali'):
            xo.assert_allclose(getattr(res, name), getattr(tan, name),
                               rtol=0, atol=0)
        for kk in (2, 3, 4):
            xo.assert_allclose(res.gali[kk], tan.gali[kk], rtol=0, atol=0)
        fma = xt.chaos.compute_fma(line, _particles(ctx, z0), window)
        for name in ('qx', 'qy', 'qx2', 'qy2', 'tune_diffusion', 'fma_valid'):
            xo.assert_allclose(getattr(res, name), getattr(fma, name),
                               rtol=0, atol=0)
        rem = xt.chaos.compute_rem(line, _particles(ctx, z0), sample_turns)
        xo.assert_allclose(res.rem, rem.rem, rtol=0, atol=0)
        assert np.all(res.rem_turns == sample_turns)

    # FLI only: one ghost, no alignment indices, no REM/FMA
    res = xt.chaos.compute_indicators(line, _particles(ctx, z0), 100,
                                      indicators=('fli',))
    assert res.log_growth.shape == (1, len(r), 1)
    for name in ('sali', 'gali', 'rem', 'qx'):
        assert not hasattr(res, name)

    with pytest.raises(ValueError, match='Unknown'):
        xt.chaos.compute_indicators(line, _particles(ctx, z0), 100,
                                    indicators=('lyapunov_spectrum',))


# 6D ghosts on a lattice ----------------------------------------------------

def _fodo_ring_with_cavity():
    cell = [xt.Drift(length=2.5),
            xt.Multipole(knl=[0, 0.18]), xt.Multipole(knl=[0, 0, 0.5]),
            xt.Drift(length=5.0),
            xt.Multipole(knl=[0, -0.18]), xt.Multipole(knl=[0, 0, -0.8]),
            xt.Drift(length=2.5)]
    elements = []
    for _ in range(8):
        elements += [ee.copy() for ee in cell]
    line = xt.Line(elements=elements)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
    beta0 = line.particle_ref.beta0[0]
    frev = beta0 * 299792458.0 / line.get_length()
    line.append('cavity', xt.Cavity(voltage=3e5, frequency=4 * frev))
    return line


@for_all_test_contexts
@allow_kernel_compilation
def test_tangent_indicators_dim6_fodo_cavity(test_context):
    line = _fodo_ring_with_cavity()
    line.build_tracker(_context=test_context)
    ctx = test_context
    tw = line.twiss()  # 6D, with synchrotron motion
    assert tw.qs > 1e-3
    W = tw.W_matrix[0]
    metric = xt.chaos.metric_from_twiss(tw, dim=6)
    xo.assert_allclose(metric @ W, np.eye(6), rtol=0, atol=1e-9)
    co = np.array([tw.x[0], tw.px[0], tw.y[0], tw.py[0], tw.zeta[0], tw.pzeta[0]])
    names = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')

    def particles(zz):
        kw = {nn: zz[kk] for kk, nn in enumerate(names)}
        return xt.Particles(_context=ctx, p0c=1e9, mass0=xt.PROTON_MASS_EV, **kw)

    # Initial conditions in normalised coordinates (all three planes)
    a = np.array([2e-4, 8e-4, 1.5e-3])
    zn0 = np.array([a, 0 * a, a, 0 * a, 3e-4 + 0 * a, 0 * a])
    z0 = co[:, None] + W @ zn0
    num_turns = 40

    res = xt.chaos.compute_tangent_indicators(
        line, particles(z0), num_turns, [10, num_turns], n_ghosts=6,
        displacement=1e-9, metric=metric, dim=6)
    assert np.all(res.valid)

    # Oracle: one-turn Jacobians by central differences along the orbit,
    # in normalised coordinates, deviation vectors renormalised every turn
    h = 1e-8
    zz = z0.copy()
    V = np.tile(np.eye(6)[None], (len(a), 1, 1))
    log_growth = np.zeros((len(a), 6))
    out = {}
    for turn in range(1, num_turns + 1):
        disp = np.concatenate([+h * W, -h * W], axis=1)  # (6, 12)
        zz_pm = (zz[:, :, None] + disp[:, None, :]).reshape(6, -1)
        p = particles(np.concatenate([zz, zz_pm], axis=1))
        line.track(p, num_turns=1)
        p.sort(interleave_lost_particles=True)
        coords = np.array([ctx.nparray_from_context_array(getattr(p, nn))
                           for nn in names])
        zz_new = coords[:, :len(a)]
        dd = coords[:, len(a):].reshape(6, len(a), 12)
        J = np.einsum('kl,lpj->pkj', metric, (dd[:, :, :6] - dd[:, :, 6:]) / (2 * h))
        V = J @ V
        r = np.linalg.norm(V, axis=1)
        log_growth += np.log(r)
        V /= r[:, None, :]
        zz = zz_new
        if turn in (10, num_turns):
            out[turn] = log_growth.copy()
    # Measured: 5e-7 (10 turns), 2e-6 (40 turns)
    for ss, turn in enumerate((10, num_turns)):
        xo.assert_allclose(res.log_growth[ss], out[turn], rtol=0, atol=1e-4)

    # Ghosts carry consistent energy variables after a renormalisation
    p_all, layout = xt.chaos.build_ghost_particles(
        particles(z0), n_ghosts=6, displacement=1e-9, metric=metric, dim=6)
    tangent = xt.chaos.GhostTangent(_context=ctx, n_ref=layout.n_ref,
                                    n_ghosts=6, dim=6, eps=1e-9, metric=metric)
    line.track(p_all, num_turns=3)
    tangent.build_slot_map(p_all)
    tangent.renormalise(p_all, ctx.zeros(1, dtype=np.float64), 0)
    pz = ctx.nparray_from_context_array(p_all.pzeta)
    check = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV, pzeta=pz)
    for nn in ('delta', 'rpp', 'rvv'):
        xo.assert_allclose(ctx.nparray_from_context_array(getattr(p_all, nn)),
                           getattr(check, nn), rtol=1e-15, atol=1e-18)


def test_serialisation_of_new_elements():
    monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=3, window=10,
                                           metric=2 * np.eye(4),
                                           closed_orbit=[1e-3, 0, 0, 0])
    henon = xt.Henonmap(omega_x=1.0, omega_y=2.0, multipole_coeffs=[2.0, -1.0],
                        norm=True)
    line = xt.Line(elements=[monitor, henon], element_names=['mon', 'henon'])
    line2 = xt.Line.from_dict(line.to_dict())
    assert isinstance(line2['mon'], xt.chaos.BirkhoffTuneMonitor)
    assert line2['mon'].window == 10 and line2['mon'].num_particles == 3
    xo.assert_allclose(line2['mon'].w_inv, 2 * np.eye(4).ravel(), rtol=0, atol=0)
    xo.assert_allclose(line2['mon'].weights, monitor.weights, rtol=0, atol=0)
    xo.assert_allclose(line2['henon'].fx_coeffs, henon.fx_coeffs, rtol=0, atol=0)
    xo.assert_allclose(line2['henon'].omega_y, 2.0, rtol=0, atol=1e-15)
