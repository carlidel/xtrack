# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Helpers shared by the Henon map and chaos indicator tests.

All lines built through :func:`context_and_line` share one tracking kernel per
test context, compiled once for a warm-up line that contains every element
class listed in ``_warmup_elements``. Lines may contain any subset of these
classes. This keeps the JIT compilation time of the test modules low.
"""

import numpy as np

import xtrack as xt

_SHARED = {}


def _warmup_elements():
    return [xt.Henonmap(), xt.LimitRect()]


def shared_context(test_context):
    """Return the context that holds the shared kernel for `test_context`."""
    key = str(test_context)
    if key not in _SHARED:
        elements = _warmup_elements()
        line = xt.Line(elements=elements,
                       element_names=[f'w{ii}' for ii in range(len(elements))])
        line.build_tracker(_context=test_context)
        _SHARED[key] = (test_context, line.tracker.track_kernel)
    return _SHARED[key][0]


def context_and_line(test_context, elements, element_names=None):
    """Build a line on the shared context, reusing the shared kernel."""
    context = shared_context(test_context)
    track_kernel = _SHARED[str(test_context)][1]
    if element_names is None:
        element_names = [f'e{ii}' for ii in range(len(elements))]
    line = xt.Line(elements=elements, element_names=element_names)
    line.build_tracker(_context=context, track_kernel=track_kernel)
    return context, line


def sorted_coords(context, particles, names=('x', 'px', 'y', 'py')):
    """Coordinates as a (len(names), N) host array ordered by particle_id."""
    pid = context.nparray_from_context_array(particles.particle_id)
    order = np.argsort(pid)
    return np.array([context.nparray_from_context_array(
        getattr(particles, nn))[order] for nn in names])
