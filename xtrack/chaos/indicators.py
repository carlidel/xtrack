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
        Phase-space dimension: 4 for ``(x, px, y, py)``, 2 for ``(x, px)``.
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
        4 for ``(x, px, y, py)``, 2 for ``(x, px)``.

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
