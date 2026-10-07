# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

from .ghosts import build_ghost_particles
from .tangent import GhostTangent


def birkhoff_weights(n):
    """Weights ``w_k``, ``k = 1..n``, of a weighted Birkhoff average.

    ``w_k ∝ exp(-1 / (t_k (1 - t_k)))`` with ``t_k = k / (n + 1)``,
    normalised to ``sum(w) = 1``.
    """
    n = int(n)
    if n < 1:
        raise ValueError('`n` must be positive')
    t = np.arange(1, n + 1) / (n + 1)
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return w / w.sum()


def renormalisation_turns(num_turns, renorm_every, sample_turns):
    """Turns after which the ghosts are renormalised: every `renorm_every`
    turns, all `sample_turns` and `num_turns`."""
    turns = set(range(int(renorm_every), int(num_turns) + 1, int(renorm_every)))
    turns |= set(int(tt) for tt in sample_turns)
    turns.add(int(num_turns))
    return np.array(sorted(turns), dtype=np.int64)


def _chunk_weights(turns, sample_turns):
    """Birkhoff weight of each renormalisation chunk for each horizon.

    For a chunk covering turns ``t0 + 1 .. t1`` and horizon ``n``, the weight
    is the mean of ``w_n`` over the chunk if ``t1 <= n``, 0 otherwise. With
    one-turn chunks this gives exactly ``sum_k w_k log(r_k / eps)``.
    """
    out = np.zeros((len(turns), len(sample_turns)))
    for ss, nn in enumerate(sample_turns):
        w = birkhoff_weights(nn)
        t0 = 0
        for cc, t1 in enumerate(turns):
            if t1 <= nn:
                out[cc, ss] = w[t0:t1].mean()
            t0 = t1
    return out


class ChaosIndicators:
    """Results of the chaos indicators, as host NumPy arrays.

    Per-orbit arrays are ordered by the original reference ``particle_id``
    (stored in ``particle_id``). Arrays with a leading axis of length
    ``len(sample_turns)`` hold the value at each sample turn; entries of
    orbits that were invalid (lost) at that turn are NaN.

    Attributes (present when the corresponding indicator was computed)
    ------------------------------------------------------------------
    particle_id : (N,) int
    sample_turns : (S,) int
    valid : (S, N) bool
        Orbit (reference and all its ghosts) still alive at the sample turn.
    lost_at_turn : (N,) int
        Earliest turn at which the reference or one of its ghosts was lost,
        -1 if never lost.
    fli : (S, N)
        Fast Lyapunov indicator ``sum log(r_k / eps)`` of ghost 0, i.e. the
        log of the stretching of a deviation vector after n turns.
    lyapunov : (S, N)
        Finite-time Lyapunov exponent estimate ``fli / n``.
    fli_birkhoff : (S, N)
        ``sum_k w_k log(r_k / eps)`` with Birkhoff weights for horizon ``n``
        (a weighted-average Lyapunov exponent per turn).
    log_growth : (S, N, G)
        ``sum log(r_k / eps)`` for every ghost.
    sali : (S, N)
        ``min(|u_0 + u_1|, |u_0 - u_1|)`` of the unit deviation vectors.
    gali : dict k -> (S, N)
        Generalised alignment index of order ``k = 2..G``, the volume spanned
        by the first k unit deviation vectors.
    rem_turns : (H,) int
        Horizons of the reversibility error.
    rem : (H, N)
        Reversibility error ``|M^-n M^n z - z|`` in the metric; NaN if the
        particle was lost. ``rem_valid`` (H, N) flags finite entries.
    qx, qy, qx2, qy2 : (N,)
        Tunes in the first and second FMA window (NaN if lost).
    tune_diffusion : (N,)
        ``log10(sqrt((qx - qx2)^2 + (qy - qy2)^2))``. ``fma_valid`` (N,)
        flags complete measurements, ``fma_method`` says how they were made.
    """

    def __init__(self, particle_id, sample_turns):
        self.particle_id = np.asarray(particle_id)
        self.sample_turns = np.asarray(sample_turns, dtype=np.int64)

    def __repr__(self):
        names = [kk for kk in self.__dict__ if not kk.startswith('_')]
        return f'ChaosIndicators({", ".join(names)})'


def _check_line(line):
    if line.tracker is None:
        raise ValueError('The line has no tracker, call `line.build_tracker()`')
    if line.tracker.iscollective:
        raise NotImplementedError('Collective lines are not supported')


def _sample_turns_or_default(num_turns, sample_turns):
    if sample_turns is None:
        sample_turns = [num_turns]
    sample_turns = np.unique(np.asarray(sample_turns, dtype=np.int64))
    if sample_turns[0] < 1 or sample_turns[-1] > num_turns:
        raise ValueError('`sample_turns` must be between 1 and `num_turns`')
    return sample_turns


def compute_tangent_indicators(line, particles, num_turns, sample_turns=None,
                               n_ghosts=4, displacement=1e-8, renorm_every=1,
                               metric=None, dim=4, alignment=True,
                               _result=None):
    """Tangent-map chaos indicators from ghost (shadow) particles.

    Every reference particle is tracked together with `n_ghosts` ghosts
    displaced by `displacement` along orthogonal directions of the metric.
    After every `renorm_every` turns (and at every sample turn) the pair
    kernels measure, for each reference, the metric distances ``r_g`` of its
    ghosts, accumulate ``log(r_g / eps)`` and move the ghosts back to
    distance ``eps`` along the same directions (no orthogonalisation).

    Parameters
    ----------
    line : xtrack.Line
        Line with a tracker; one pass through the line is one turn.
    particles : xtrack.Particles
        Reference particles (all alive). They are not modified.
    num_turns : int
        Number of turns.
    sample_turns : sequence of int, optional
        Turns at which the indicators are returned. Default ``[num_turns]``.
    n_ghosts : int
        Ghosts per reference (1 for FLI only, >= 2 for SALI/GALI).
    displacement : float
        Ghost distance ``eps`` in the metric. The finite-difference floor of
        the tangent estimate is ``~1e-16 |z| / eps``.
    renorm_every : int
        Renormalisation period in turns. With ``renorm_every > 1`` the
        Birkhoff FLI uses chunk-averaged weights (exact for 1).
    metric : (dim, dim) array, optional
        Maps a phase-space displacement to the space where norms are taken
        (identity by default; see :func:`metric_from_twiss`).
    dim : int
        Phase-space dimension: 4 for ``(x, px, y, py)``, 2 for ``(x, px)``,
        6 for ``(x, px, y, py, zeta, pzeta)`` (energy variables of the
        ghosts are updated consistently from ``pzeta``).
    alignment : bool
        Compute SALI and GALI (needs ``n_ghosts >= 2``).

    Returns
    -------
    ChaosIndicators
        With ``fli``, ``lyapunov``, ``fli_birkhoff``, ``log_growth``,
        ``valid``, ``lost_at_turn`` and, if `alignment`, ``sali`` and
        ``gali``.
    """
    _check_line(line)
    if n_ghosts < 1:
        raise ValueError('At least one ghost is needed')
    context = line._context
    num_turns = int(num_turns)
    sample_turns = _sample_turns_or_default(num_turns, sample_turns)
    n_samples = len(sample_turns)
    flag_alignment = bool(alignment) and n_ghosts >= 2

    p_all, layout = build_ghost_particles(
        particles, n_ghosts=n_ghosts, displacement=displacement,
        metric=metric, dim=dim, _context=context)
    n_ref = layout.n_ref

    tangent = GhostTangent(_context=context, n_ref=n_ref, n_ghosts=n_ghosts,
                           dim=dim, eps=displacement, metric=metric,
                           n_samples=n_samples)

    turns = renormalisation_turns(num_turns, renorm_every, sample_turns)
    weights = context.nparray_to_context_array(
        np.ascontiguousarray(_chunk_weights(turns, sample_turns).ravel()))

    res = _result or ChaosIndicators(layout.reference_particle_id, sample_turns)
    valid = np.zeros((n_samples, n_ref), dtype=bool)
    log_growth = np.full((n_samples, n_ref, n_ghosts), np.nan)
    fli_wb = np.full((n_samples, n_ref), np.nan)
    sali = np.full((n_samples, n_ref), np.nan)
    gali = np.full((n_samples, n_ref, max(n_ghosts - 1, 0)), np.nan)

    to_host = context.nparray_from_context_array
    sample_index = {int(tt): ss for ss, tt in enumerate(sample_turns)}
    turn = 0
    for chunk_index, t_next in enumerate(turns):
        line.track(p_all, num_turns=int(t_next - turn))
        tangent.build_slot_map(p_all)
        tangent.renormalise(p_all, weights, chunk_index, flag_alignment)
        turn = int(t_next)
        if turn in sample_index:
            ss = sample_index[turn]
            vv = to_host(tangent.valid)[:n_ref].astype(bool)
            valid[ss] = vv
            lg = to_host(tangent.log_growth)[:n_ref * n_ghosts].reshape(
                n_ref, n_ghosts)
            log_growth[ss][vv] = lg[vv]
            fw = to_host(tangent.fli_wb)[:n_ref * n_samples].reshape(
                n_ref, n_samples)
            fli_wb[ss][vv] = fw[vv, ss]
            if flag_alignment:
                sali[ss][vv] = to_host(tangent.sali)[:n_ref][vv]
                gg = to_host(tangent.gali)[:n_ref * (n_ghosts - 1)].reshape(
                    n_ref, n_ghosts - 1)
                gali[ss][vv] = gg[vv]

    res.valid = valid
    res.lost_at_turn = to_host(tangent.lost_at_turn)[:n_ref].copy()
    res.log_growth = log_growth
    res.fli = log_growth[:, :, 0].copy()
    res.lyapunov = res.fli / sample_turns[:, None]
    res.fli_birkhoff = fli_wb
    if flag_alignment:
        res.sali = sali
        res.gali = {kk: gali[:, :, kk - 2].copy() for kk in range(2, n_ghosts + 1)}
    return res


def _check_backtrackable(line):
    _check_line(line)
    if line.enable_time_dependent_vars:
        raise ValueError('REM is not supported with time-dependent variables '
                         '(backtracking does not reverse them)')
    if not line.tracker._tracker_data_base._is_backtrackable:
        from ..line import _has_backtrack
        elements = [line._element_dict[nn] for nn in line.element_names]
        not_bt = sorted({type(ee).__name__ for ee in elements
                         if not _has_backtrack(ee, line)})
        raise ValueError('REM needs a backtrackable line; elements without '
                         f'backtrack: {not_bt}')


def compute_rem(line, particles, rem_turns, metric=None, dim=4,
                _result=None):
    """Reversibility error method (REM).

    For every horizon ``n`` in `rem_turns` the particles are tracked ``n``
    turns forward and then ``n`` turns backward (``backtrack=True``);
    ``REM(n)`` is the metric distance between the final and the initial
    coordinates. A single forward pass is used: at each horizon a copy of
    the particles is backtracked. REM is driven by round-off, so values are
    meaningful as orders of magnitude (regular orbits: near round-off,
    growing polynomially; chaotic orbits: growing exponentially, up to the
    size of the orbit).

    Parameters
    ----------
    line : xtrack.Line
        Backtrackable, non-collective line without time-dependent variables.
    particles : xtrack.Particles
        Reference particles (all alive). They are not modified.
    rem_turns : int or sequence of int
        Horizons ``n``.
    metric : (dim, dim) array, optional
        Metric for the distance (identity by default).
    dim : int
        4 for ``(x, px, y, py)``, 2 for ``(x, px)``, 6 for
        ``(x, px, y, py, zeta, pzeta)``.

    Returns
    -------
    ChaosIndicators
        With ``rem_turns`` (H,), ``rem`` (H, N) (NaN where the particle was
        lost in either direction) and ``rem_valid`` (H, N).
    """
    _check_backtrackable(line)
    context = line._context
    rem_turns = np.unique(np.atleast_1d(np.asarray(rem_turns, dtype=np.int64)))
    if rem_turns[0] < 1:
        raise ValueError('`rem_turns` must be positive')

    p_ref, layout = build_ghost_particles(particles, n_ghosts=0, metric=metric,
                                          dim=dim, _context=context)
    n_ref = layout.n_ref
    tangent = GhostTangent(_context=context, n_ref=n_ref, n_ghosts=0, dim=dim,
                           metric=metric)
    to_host = context.nparray_from_context_array
    from .ghosts import COORD_NAMES
    pid = to_host(p_ref.particle_id)
    coords0 = np.zeros((n_ref, dim))
    for kk in range(dim):
        coords0[pid, kk] = to_host(getattr(p_ref, COORD_NAMES[kk]))
    coords0 = context.nparray_to_context_array(np.ascontiguousarray(coords0.ravel()))
    out = context.zeros(n_ref, dtype=np.float64)

    rem = np.full((len(rem_turns), n_ref), np.nan)
    turn = 0
    for hh, n_turns in enumerate(rem_turns):
        line.track(p_ref, num_turns=int(n_turns - turn))
        turn = int(n_turns)
        p_back = p_ref.copy()
        line.track(p_back, num_turns=int(n_turns), backtrack=True)
        tangent.distance(p_back, coords0, out)
        dd = to_host(out).copy()
        rem[hh] = np.where(dd >= 0, dd, np.nan)
        del p_back

    res = _result or ChaosIndicators(layout.reference_particle_id, rem_turns)
    res.rem_turns = rem_turns
    res.rem = rem
    res.rem_valid = np.isfinite(rem)
    return res


def _tune_monitor_names(line):
    from .tune_monitor import BirkhoffTuneMonitor
    return [nn for nn in line.element_names
            if isinstance(line._element_dict[nn], BirkhoffTuneMonitor)]


def _get_inline_tune_monitor(line, n_ref, window, start_turn):
    names = _tune_monitor_names(line)
    if len(names) == 0:
        return None
    if len(names) > 1:
        raise ValueError(f'More than one BirkhoffTuneMonitor in the line: {names}')
    mon = line._element_dict[names[0]]
    if (mon.particle_id_start != 0 or mon.num_particles < n_ref
            or mon.window != window or mon.start_turn != start_turn):
        raise ValueError(
            f'The BirkhoffTuneMonitor `{names[0]}` in the line must have '
            f'particle_id_start=0, num_particles>={n_ref}, window={window} '
            f'and start_turn={start_turn}')
    return mon


def compute_fma(line, particles, window, start_turn=0, metric=None,
                closed_orbit=None, method='birkhoff', _result=None):
    """Frequency map analysis: tunes in two consecutive windows and their
    diffusion ``log10(sqrt(dqx^2 + dqy^2))``.

    Parameters
    ----------
    line : xtrack.Line
        Line with a tracker (one pass is one turn).
    particles : xtrack.Particles
        Reference particles (all alive). They are not modified.
    window : int
        Turns per window; ``start_turn + 2 window`` turns are tracked.
    start_turn : int
        Turn at which the first window starts.
    metric, closed_orbit :
        Normalisation ``z_n = metric (z - closed_orbit)`` of
        ``(x, px, y, py)``; identity and zero by default (use
        :func:`metric_from_twiss` and the twiss closed orbit for a lattice).
    method : {'birkhoff', 'naff'}
        ``'birkhoff'`` (default): Birkhoff-weighted phase advances
        accumulated in a kernel by :class:`BirkhoffTuneMonitor`, on any
        context. If the line contains a ``BirkhoffTuneMonitor`` (with
        ``particle_id_start=0``, ``num_particles >= N`` and the same
        `window` and `start_turn`) it is used and all turns are tracked in a
        single call; otherwise a standalone monitor is applied between
        one-turn tracking calls. ``'naff'``: turn-by-turn data and
        ``nafflib.tune`` on the host (reference path; needs ``nafflib``).
        The Birkhoff tune is a rotation number around the origin: it
        agrees with NAFF when the projected motion winds around the origin,
        but not when a plane is dominated by other frequencies (see
        :class:`BirkhoffTuneMonitor`).

    Returns
    -------
    ChaosIndicators
        With ``qx``, ``qy`` (first window), ``qx2``, ``qy2`` (second
        window), ``tune_diffusion`` and ``fma_valid``; NaN for particles lost
        before the end of the second window.
    """
    from .tune_monitor import BirkhoffTuneMonitor
    _check_line(line)
    context = line._context
    window = int(window)
    start_turn = int(start_turn)
    if method not in ('birkhoff', 'naff'):
        raise ValueError("`method` must be 'birkhoff' or 'naff'")

    p_ref, layout = build_ghost_particles(particles, n_ghosts=0, _context=context)
    n_ref = layout.n_ref
    metric = np.eye(4) if metric is None else np.asarray(metric, dtype=float)
    closed_orbit = (np.zeros(4) if closed_orbit is None
                    else np.asarray(closed_orbit, dtype=float))

    if method == 'birkhoff':
        mon = _get_inline_tune_monitor(line, n_ref, window, start_turn)
        if mon is not None:
            mon.w_inv[:] = context.nparray_to_context_array(metric.ravel())
            mon.closed_orbit[:] = context.nparray_to_context_array(closed_orbit)
            mon.reset()
            line.track(p_ref, num_turns=start_turn + 2 * window + 1)
        else:
            mon = BirkhoffTuneMonitor(
                _context=context, num_particles=n_ref, window=window,
                start_turn=start_turn, metric=metric, closed_orbit=closed_orbit)
            if start_turn > 0:
                line.track(p_ref, num_turns=start_turn)
            for jj in range(2 * window + 1):
                mon.track(p_ref)
                if jj < 2 * window:
                    line.track(p_ref, num_turns=1)
        qx1, qy1, qx2, qy2, diffusion = (vv[:n_ref] for vv in mon.get_tunes())
    else:
        import nafflib
        n_turns = start_turn + 2 * window
        line.track(p_ref, num_turns=n_turns, turn_by_turn_monitor=True)
        rec = line.record_last_track
        coords = np.array([rec.x, rec.px, rec.y, rec.py])[:, :n_ref, :]
        zn = np.einsum('kl,lpt->kpt', metric, coords - closed_orbit[:, None, None])
        state = context.nparray_from_context_array(p_ref.state)
        pid = context.nparray_from_context_array(p_ref.particle_id)
        alive = np.zeros(n_ref, dtype=bool)
        alive[pid[pid < n_ref]] = state[pid < n_ref] > 0
        qq = np.full((4, n_ref), np.nan)
        w1 = slice(start_turn, start_turn + window)
        w2 = slice(start_turn + window, start_turn + 2 * window)
        for ii in np.where(alive)[0]:
            qq[0, ii] = nafflib.tune(zn[0, ii, w1], zn[1, ii, w1]) % 1
            qq[1, ii] = nafflib.tune(zn[2, ii, w1], zn[3, ii, w1]) % 1
            qq[2, ii] = nafflib.tune(zn[0, ii, w2], zn[1, ii, w2]) % 1
            qq[3, ii] = nafflib.tune(zn[2, ii, w2], zn[3, ii, w2]) % 1
        qx1, qy1, qx2, qy2 = qq
        with np.errstate(divide='ignore', invalid='ignore'):
            diffusion = np.log10(np.sqrt((qx1 - qx2)**2 + (qy1 - qy2)**2))

    res = _result or ChaosIndicators(layout.reference_particle_id,
                                     [start_turn + 2 * window])
    res.qx, res.qy, res.qx2, res.qy2 = qx1, qy1, qx2, qy2
    res.tune_diffusion = diffusion
    res.fma_valid = np.isfinite(qx1)
    res.fma_method = method
    return res


ALL_INDICATORS = ('fli', 'sali', 'gali', 'rem', 'fma')


def compute_indicators(line, particles, num_turns, sample_turns=None,
                       indicators=ALL_INDICATORS, n_ghosts=4,
                       displacement=1e-8, renorm_every=1, metric=None,
                       rem_turns=None, fma_window=None, fma_start_turn=0,
                       closed_orbit=None, fma_method='birkhoff', dim=4):
    """Compute several chaos indicators for the same reference particles.

    Parameters
    ----------
    line : xtrack.Line
        Line with a tracker; one pass is one turn.
    particles : xtrack.Particles
        Reference particles (all alive). They are not modified.
    num_turns : int
        Turns of the tangent (ghost) run.
    sample_turns : sequence of int, optional
        Turns at which the tangent indicators are returned (default
        ``[num_turns]``); also the default REM horizons.
    indicators : sequence of str
        Any of ``'fli'`` (FLI, Lyapunov estimate and Birkhoff FLI),
        ``'sali'``, ``'gali'``, ``'rem'``, ``'fma'``.
    n_ghosts : int
        Ghosts per reference for SALI/GALI (one ghost is used if only FLI
        is requested).
    displacement, renorm_every, metric, dim :
        See :func:`compute_tangent_indicators`. The metric and `dim` are
        also used by REM; FMA always works in ``(x, px, y, py)`` and takes
        the upper-left 4x4 block of the metric.
    rem_turns : sequence of int, optional
        REM horizons (default `sample_turns`).
    fma_window, fma_start_turn, closed_orbit, fma_method :
        See :func:`compute_fma`. Default window
        ``(num_turns - fma_start_turn - 1) // 2``.

    Notes
    -----
    Passes over the line: one ghost pass for the tangent indicators; REM
    needs its own forward + backward pass. FMA is measured within the ghost
    pass when the line contains a matching :class:`BirkhoffTuneMonitor`
    (``particle_id_start=0``, ``num_particles >= N``, same window and start
    turn) and ``fma_method='birkhoff'``; otherwise it uses a separate pass.

    Returns
    -------
    ChaosIndicators
        All requested results, ordered by reference ``particle_id``.
    """
    indicators = tuple(indicators)
    unknown = set(indicators) - set(ALL_INDICATORS)
    if unknown:
        raise ValueError(f'Unknown indicators: {sorted(unknown)}')
    _check_line(line)
    num_turns = int(num_turns)
    sample_turns = _sample_turns_or_default(num_turns, sample_turns)
    if 'rem' in indicators:
        _check_backtrackable(line)  # fail before the long passes

    p_sorted, layout = build_ghost_particles(particles, n_ghosts=0,
                                             _context=line._context)
    res = ChaosIndicators(layout.reference_particle_id, sample_turns)
    res.indicators = indicators
    del p_sorted

    want_tangent = any(ii in indicators for ii in ('fli', 'sali', 'gali'))
    want_alignment = 'sali' in indicators or 'gali' in indicators

    fma_done = False
    if 'fma' in indicators:
        if fma_window is None:
            fma_window = (num_turns - fma_start_turn - 1) // 2
        if fma_window < 1:
            raise ValueError('`num_turns` is too short for the FMA windows')

    if want_tangent:
        mon = None
        if ('fma' in indicators and fma_method == 'birkhoff'
                and fma_start_turn + 2 * fma_window + 1 <= num_turns):
            mon = _get_inline_tune_monitor(line, layout.n_ref, fma_window,
                                           fma_start_turn)
            if mon is not None:
                ctx = line._context
                mm = np.eye(4) if metric is None else np.asarray(metric)[:4, :4]
                co = np.zeros(4) if closed_orbit is None else np.asarray(closed_orbit)
                mon.w_inv[:] = ctx.nparray_to_context_array(
                    np.ascontiguousarray(mm, dtype=float).ravel())
                mon.closed_orbit[:] = ctx.nparray_to_context_array(
                    np.asarray(co, dtype=float))
                mon.reset()
        compute_tangent_indicators(
            line, particles, num_turns, sample_turns,
            n_ghosts=n_ghosts if want_alignment else 1,
            displacement=displacement, renorm_every=renorm_every,
            metric=metric, dim=dim, alignment=want_alignment, _result=res)
        if mon is not None:
            qx1, qy1, qx2, qy2, diff = (vv[:layout.n_ref]
                                        for vv in mon.get_tunes())
            res.qx, res.qy, res.qx2, res.qy2 = qx1, qy1, qx2, qy2
            res.tune_diffusion = diff
            res.fma_valid = np.isfinite(qx1)
            res.fma_method = 'birkhoff (in-line, ghost pass)'
            fma_done = True
        if 'sali' not in indicators and hasattr(res, 'sali'):
            del res.sali
        if 'gali' not in indicators and hasattr(res, 'gali'):
            del res.gali

    if 'fma' in indicators and not fma_done:
        compute_fma(line, particles, fma_window, start_turn=fma_start_turn,
                    metric=None if metric is None else np.asarray(metric)[:4, :4],
                    closed_orbit=closed_orbit,
                    method=fma_method, _result=res)

    if 'rem' in indicators:
        compute_rem(line, particles,
                    sample_turns if rem_turns is None else rem_turns,
                    metric=metric, dim=dim, _result=res)
    return res
