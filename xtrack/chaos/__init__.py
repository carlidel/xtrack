# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Chaos indicators computed in xobjects kernels.

Tangent indicators (FLI, Birkhoff-weighted FLI, SALI, GALI) use ghost
(shadow) particles tracked together with the reference particles; pair
kernels between tracking calls measure and renormalise their separations.
"""

from .ghosts import GhostLayout, build_ghost_particles, metric_from_twiss
from .tangent import GhostTangent
from .indicators import (
    ChaosIndicators,
    birkhoff_weights,
    compute_tangent_indicators,
    renormalisation_turns,
)
