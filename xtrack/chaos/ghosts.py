# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np

import xobjects as xo

from ..particles import Particles

COORD_NAMES = ('x', 'px', 'y', 'py', 'zeta', 'pzeta')


class GhostLayout:
    """Particle-id layout of references and ghosts.

    Reference ``i`` (``0 <= i < n_ref``) has ``particle_id = i``; ghost ``g``
    of reference ``i`` has ``particle_id = n_ref + i * n_ghosts + g`` and
    ``parent_particle_id = i``. ``reference_particle_id[i]`` is the
    particle_id the reference had in the particles passed by the user.
    """

    def __init__(self, n_ref, n_ghosts, reference_particle_id):
        self.n_ref = n_ref
        self.n_ghosts = n_ghosts
        self.reference_particle_id = np.asarray(reference_particle_id)

    @property
    def num_particles(self):
        return self.n_ref * (self.n_ghosts + 1)

    def ghost_id(self, i_ref, g):
        return self.n_ref + np.asarray(i_ref) * self.n_ghosts + g


def metric_from_twiss(twiss, dim=4, at=0):
    """Metric taking a phase-space displacement to normalised coordinates,
    i.e. the upper-left ``dim x dim`` block of ``W^-1`` at index `at`."""
    W = np.asarray(twiss.W_matrix[at])
    return np.linalg.inv(W)[:dim, :dim]


def _host_particles_sorted(particles):
    """Host copy of the particles, sorted by particle_id, without unused
    slots. All particles must be alive."""
    p_host = particles.copy(_context=xo.context_default)
    p_host = p_host.remove_unused_space()
    if np.any(p_host.state <= 0):
        raise ValueError('All reference particles must be alive (state > 0)')
    if len(np.unique(p_host.particle_id)) != len(p_host.particle_id):
        raise ValueError('Reference particle ids must be unique')
    dct = p_host.to_dict(remove_underscored=True, keep_rng_state=False)
    order = np.argsort(p_host.particle_id, kind='stable')
    n_part = len(order)
    out = {}
    for kk, vv in dct.items():
        if isinstance(vv, np.ndarray) and vv.shape == (n_part,):
            out[kk] = vv[order]
        else:
            out[kk] = vv
    return out, n_part


def build_ghost_particles(particles, n_ghosts=4, displacement=1e-8,
                          metric=None, dim=4, _context=None):
    """Return ``(particles_with_ghosts, layout)``.

    The returned :class:`xtrack.Particles` holds the references (sorted by
    their original particle_id, renumbered ``0..N-1``) followed by
    ``n_ghosts`` ghosts per reference (see :class:`GhostLayout`). Ghost ``g``
    is displaced from its reference by ``displacement * metric^-1 e_g``, so
    that in the metric the displacements are orthogonal with norm
    `displacement`. Coordinates are ``(x, px)`` (dim 2), ``(x, px, y, py)``
    (dim 4) or ``(x, px, y, py, zeta, pzeta)`` (dim 6, the order of the
    twiss ``W_matrix``). The reference particles passed in are not modified.
    """
    if dim not in (2, 4, 6):
        raise ValueError('`dim` must be 2, 4 or 6')
    if not 0 <= n_ghosts <= dim:
        raise ValueError('`n_ghosts` must be between 0 and `dim`')
    if _context is None:
        _context = particles._context

    metric = np.eye(dim) if metric is None else np.asarray(metric, dtype=float)
    if metric.shape != (dim, dim):
        raise ValueError(f'`metric` must have shape ({dim}, {dim})')
    directions = np.linalg.inv(metric)  # column g: physical direction of e_g

    dct, n_ref = _host_particles_sorted(particles)
    reference_particle_id = dct['particle_id'].copy()

    idx = np.concatenate([np.arange(n_ref), np.repeat(np.arange(n_ref), n_ghosts)])
    n_tot = len(idx)
    for kk, vv in list(dct.items()):
        if isinstance(vv, np.ndarray) and vv.shape == (n_ref,):
            dct[kk] = vv[idx].copy()

    dct['particle_id'] = np.arange(n_tot, dtype=np.int64)
    dct['parent_particle_id'] = idx.astype(np.int64)

    gg = np.tile(np.arange(n_ghosts), n_ref)
    for kk in range(dim):
        name = COORD_NAMES[kk]
        dct[name][n_ref:] += displacement * directions[kk, gg]
    if dim == 6:
        # delta, rpp and rvv are recomputed from pzeta
        for kk in ('delta', 'rpp', 'rvv'):
            dct.pop(kk, None)

    out = Particles.from_dict(dct, load_rng_state=False, _context=_context)
    layout = GhostLayout(n_ref=n_ref, n_ghosts=n_ghosts,
                         reference_particle_id=reference_particle_id)
    return out, layout
