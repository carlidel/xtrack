# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo

from ..particles import Particles

MAX_DIM = 6


class GhostTangent(xo.HybridClass):
    """Device-side state of the ghost-particle tangent indicators.

    Holds, per reference orbit, the accumulators updated by the pair kernels
    of ``chaos/src/ghost_tangent.h`` between tracking calls. Reference ``i``
    has ``particle_id = i`` and its ghost ``g`` has
    ``particle_id = n_ref + i * n_ghosts + g`` (see
    :func:`xtrack.chaos.build_ghost_particles`).

    Fields
    ------
    n_ref, n_ghosts, dim, n_samples : int
        Number of reference orbits, ghosts per reference, phase-space
        dimension and number of sample turns (Birkhoff FLI horizons).
    eps : float
        Distance of the ghosts from their reference in the metric.
    metric : float[36]
        ``dim x dim`` matrix (row major) mapping a phase-space displacement
        to the space in which norms are taken (e.g. normalised coordinates).
    slot_of_id : int[n_ref * (n_ghosts + 1)]
        Slot of each particle_id in the Particles arrays.
    valid : int8[n_ref]
        0 once the reference or any of its ghosts is lost; outputs are frozen.
    lost_at_turn : int[n_ref]
        Earliest ``at_turn`` among the lost particles of an invalid orbit, -1
        otherwise.
    log_growth : float[n_ref * n_ghosts]
        ``sum log(r_g / eps)`` over renormalisations, per ghost.
    fli_wb : float[n_ref * n_samples]
        Birkhoff-weighted ``sum w log(r_0 / eps)``, one per horizon.
    sali : float[n_ref]
        SALI at the last renormalisation.
    gali : float[n_ref * (n_ghosts - 1)]
        GALI_k, k = 2..n_ghosts, at the last renormalisation.
    """

    _xofields = {
        'n_ref': xo.Int64,
        'n_ghosts': xo.Int64,
        'dim': xo.Int64,
        'n_samples': xo.Int64,
        'eps': xo.Float64,
        'metric': xo.Float64[MAX_DIM * MAX_DIM],
        'slot_of_id': xo.Int64[:],
        'valid': xo.Int8[:],
        'lost_at_turn': xo.Int64[:],
        'log_growth': xo.Float64[:],
        'fli_wb': xo.Float64[:],
        'sali': xo.Float64[:],
        'gali': xo.Float64[:],
    }

    _depends_on = [Particles]

    # Not part of the prebuilt kernels
    allow_kernel_compilation = True

    _extra_c_sources = [
        '#include "xtrack/chaos/src/ghost_tangent.h"',
    ]

    _kernels = {
        'GhostTangent_build_slot_map': xo.Kernel(
            c_name='GhostTangent_build_slot_map',
            args=[
                xo.Arg(xo.ThisClass, name='state'),
                xo.Arg(Particles._XoStruct, name='particles'),
                xo.Arg(xo.Int64, name='n_slots'),
            ],
            n_threads='n_slots'),
        'GhostTangent_renormalise': xo.Kernel(
            c_name='GhostTangent_renormalise',
            args=[
                xo.Arg(xo.ThisClass, name='state'),
                xo.Arg(Particles._XoStruct, name='particles'),
                xo.Arg(xo.Float64, pointer=True, name='weights'),
                xo.Arg(xo.Int64, name='chunk_index'),
                xo.Arg(xo.Int64, name='flag_gali'),
                xo.Arg(xo.Int64, name='n_ref'),
            ],
            n_threads='n_ref'),
        'GhostTangent_distance': xo.Kernel(
            c_name='GhostTangent_distance',
            args=[
                xo.Arg(xo.ThisClass, name='state'),
                xo.Arg(Particles._XoStruct, name='particles'),
                xo.Arg(xo.Float64, pointer=True, name='coords0'),
                xo.Arg(xo.Float64, pointer=True, name='out'),
                xo.Arg(xo.Int64, name='n_ids'),
                xo.Arg(xo.Int64, name='n_slots'),
            ],
            n_threads='n_slots'),
    }

    def __init__(self, n_ref=None, n_ghosts=0, dim=4, eps=1e-8, metric=None,
                 n_samples=1, **kwargs):
        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            super().__init__(**kwargs)
            return

        if dim not in (2, 4):
            raise ValueError('Only dim = 2 or 4 is supported')
        if n_ghosts < 0 or n_ghosts > dim:
            raise ValueError('`n_ghosts` must be between 0 and `dim`')

        metric = np.eye(dim) if metric is None else np.asarray(metric, dtype=float)
        if metric.shape != (dim, dim):
            raise ValueError(f'`metric` must have shape ({dim}, {dim})')
        metric_flat = np.zeros(MAX_DIM * MAX_DIM)
        metric_flat[:dim * dim] = metric.ravel()

        # Arrays are never empty (portability of the generated structs)
        super().__init__(
            n_ref=n_ref,
            n_ghosts=n_ghosts,
            dim=dim,
            n_samples=n_samples,
            eps=eps,
            metric=metric_flat,
            slot_of_id=np.zeros(max(1, n_ref * (n_ghosts + 1)), dtype=np.int64),
            valid=np.ones(n_ref, dtype=np.int8),
            lost_at_turn=-np.ones(n_ref, dtype=np.int64),
            log_growth=np.zeros(max(1, n_ref * n_ghosts)),
            fli_wb=np.zeros(max(1, n_ref * n_samples)),
            sali=np.full(n_ref, np.sqrt(2.0)),
            gali=np.ones(max(1, n_ref * (n_ghosts - 1))),
            **kwargs)

    # Kernel wrappers -------------------------------------------------------
    def _kernel(self, name):
        context = self._context
        if name not in context.kernels:
            self.compile_kernels(only_if_needed=True)
        return context.kernels[name]

    def build_slot_map(self, particles):
        self._kernel('GhostTangent_build_slot_map')(
            state=self, particles=particles, n_slots=particles._capacity)

    def renormalise(self, particles, weights, chunk_index, flag_gali=True):
        self._kernel('GhostTangent_renormalise')(
            state=self, particles=particles, weights=weights,
            chunk_index=int(chunk_index), flag_gali=int(bool(flag_gali)),
            n_ref=self.n_ref)

    def distance(self, particles, coords0, out):
        """Metric distance of each particle (by particle_id < len(out)) to
        ``coords0[particle_id * dim : (particle_id + 1) * dim]``; -1 if
        lost."""
        self._kernel('GhostTangent_distance')(
            state=self, particles=particles, coords0=coords0, out=out,
            n_ids=len(out), n_slots=particles._capacity)
