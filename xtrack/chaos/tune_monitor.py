# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo
import xtrack as xt

from ..base_element import BeamElement


class BirkhoffTuneMonitor(BeamElement):
    """Thin, passive element measuring tunes and tune diffusion in-line.

    For each particle with ``particle_id_start <= particle_id <
    particle_id_start + num_particles`` the element computes, at turns
    ``start_turn + j`` (``j = 0 .. 2 window``), the normalised coordinates
    ``z_n = W^-1 (z - closed_orbit)`` of ``(x, px, y, py)`` and the phases
    ``theta = atan2(-p_n, q_n)`` per plane. The phase advance of
    observation ``j >= 1``, taken in ``[0, 2 pi)``, is averaged with the
    Birkhoff weights ``w`` of :func:`xtrack.chaos.birkhoff_weights` over two
    consecutive windows of ``window`` turns, giving ``(qx1, qy1)`` and
    ``(qx2, qy2)``. The tune diffusion is
    ``log10(sqrt((qx1 - qx2)^2 + (qy1 - qy2)^2))``.

    Memory is O(1) per particle (plus ``window`` shared weights). Tunes are
    measured in ``[0, 1)``; the advance per turn must stay below ``2 pi``
    (no aliasing correction). The element is passive in backtracking.

    Place it where the normalisation applies (usually at the start of the
    line, so that at turn ``t`` it sees the state after ``t`` turns) and
    track at least ``start_turn + 2 window + 1`` turns.

    Parameters
    ----------
    num_particles : int
        Number of monitored particle ids.
    window : int
        Number of turns of each averaging window.
    start_turn : int
        Turn of the first observation. Default 0.
    particle_id_start : int
        First monitored particle id. Default 0.
    metric : (4, 4) array, optional
        ``W^-1`` restricted to ``(x, px, y, py)`` (identity by default, i.e.
        coordinates already normalised; see
        :func:`xtrack.chaos.metric_from_twiss`).
    closed_orbit : (4,) array, optional
        Closed orbit ``(x, px, y, py)`` at the element. Default zero.
    """

    _xofields = {
        'start_turn': xo.Int64,
        'window': xo.Int64,
        'particle_id_start': xo.Int64,
        'num_particles': xo.Int64,
        'w_inv': xo.Float64[16],
        'closed_orbit': xo.Float64[4],
        'weights': xo.Float64[:],
        'phase_x': xo.Float64[:],
        'phase_y': xo.Float64[:],
        'qx1': xo.Float64[:],
        'qy1': xo.Float64[:],
        'qx2': xo.Float64[:],
        'qy2': xo.Float64[:],
        'n_advances': xo.Int64[:],
    }

    isthick = False
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = True
    allow_rot_and_shift = False

    # Not part of the prebuilt kernels
    allow_kernel_compilation = True

    _extra_c_sources = [
        '#include "xtrack/chaos/src/birkhoff_tune_monitor.h"',
    ]

    _per_particle_arrays = ('phase_x', 'phase_y', 'qx1', 'qy1', 'qx2', 'qy2',
                            'n_advances')

    def __init__(self, num_particles=None, window=None, start_turn=0,
                 particle_id_start=0, metric=None, closed_orbit=None,
                 _xobject=None, **kwargs):
        if _xobject is not None:
            super().__init__(_xobject=_xobject, **kwargs)
            return

        if num_particles is None or window is None:
            raise ValueError('`num_particles` and `window` are required')
        num_particles = int(num_particles)
        window = int(window)
        if num_particles < 1 or window < 1:
            raise ValueError('`num_particles` and `window` must be positive')

        from .indicators import birkhoff_weights
        metric = np.eye(4) if metric is None else np.asarray(metric, dtype=float)
        if metric.shape != (4, 4):
            raise ValueError('`metric` must have shape (4, 4)')
        closed_orbit = (np.zeros(4) if closed_orbit is None
                        else np.asarray(closed_orbit, dtype=float))
        if closed_orbit.shape != (4,):
            raise ValueError('`closed_orbit` must have shape (4,)')

        kwargs.setdefault('phase_x', np.zeros(num_particles))
        kwargs.setdefault('phase_y', np.zeros(num_particles))
        kwargs.setdefault('qx1', np.zeros(num_particles))
        kwargs.setdefault('qy1', np.zeros(num_particles))
        kwargs.setdefault('qx2', np.zeros(num_particles))
        kwargs.setdefault('qy2', np.zeros(num_particles))
        kwargs.setdefault('n_advances', np.zeros(num_particles, dtype=np.int64))
        kwargs.setdefault('weights', birkhoff_weights(window))

        super().__init__(start_turn=start_turn, window=window,
                         particle_id_start=particle_id_start,
                         num_particles=num_particles,
                         w_inv=metric.ravel(), closed_orbit=closed_orbit,
                         **kwargs)

    def reset(self):
        """Clear the accumulators (to reuse the monitor for a new run)."""
        for nn in self._per_particle_arrays:
            getattr(self, nn)[:] = 0

    def _host(self, name):
        return self._context.nparray_from_context_array(getattr(self, name)).copy()

    @property
    def complete(self):
        """True for particles that went through both windows."""
        return self._host('n_advances') == 2 * self.window

    def get_tunes(self):
        """Return ``(qx1, qy1, qx2, qy2, diffusion)`` as host arrays indexed
        by ``particle_id - particle_id_start``; NaN for particles that did not
        complete both windows."""
        out = [self._host(nn) for nn in ('qx1', 'qy1', 'qx2', 'qy2')]
        complete = self.complete
        for vv in out:
            vv[~complete] = np.nan
        qx1, qy1, qx2, qy2 = out
        with np.errstate(divide='ignore', invalid='ignore'):
            diffusion = np.log10(np.sqrt((qx1 - qx2)**2 + (qy1 - qy2)**2))
        return qx1, qy1, qx2, qy2, diffusion

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        return xt.Marker(_context=_context, _buffer=_buffer, _offset=_offset)
