.. Draft for the xsuite documentation (docs repository), in the style of
   footprint.rst. Paths of literalinclude/image refer to the xtrack examples.

================
Chaos indicators
================

The module ``xtrack.chaos`` computes short-run indicators of chaotic motion
that are used to predict long-term stability (dynamic aperture) from a few
thousand turns: the fast Lyapunov indicator (FLI) and its Birkhoff-weighted
version, the smaller and generalised alignment indices (SALI, GALI\ :sub:`k`),
the reversibility error method (REM) and the frequency map analysis (FMA,
tune diffusion). All the per-orbit arithmetic runs in xobjects kernels, so
the same code runs on CPU (serial or OpenMP) and on GPUs (CUDA, OpenCL).

Quick start
===========

.. code-block:: python

    import numpy as np
    import xtrack as xt

    # Any line with a tracker; here the 4D Henon map (normalised coordinates)
    henon = xt.Henonmap(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201,
                        multipole_coeffs=[2.0], norm=True)
    line = xt.Line(elements=[henon, xt.LimitRect(min_x=-1, max_x=1,
                                                 min_y=-1, max_y=1)])
    line.build_tracker()

    x0, y0 = np.meshgrid(np.linspace(0.01, 0.5, 50), np.linspace(0.01, 0.5, 50))
    particles = xt.Particles(x=x0.ravel(), y=y0.ravel())

    res = xt.chaos.compute_indicators(line, particles, num_turns=2000,
                                      sample_turns=[500, 2000])
    res.fli          # (2, N) FLI at 500 and 2000 turns
    res.sali         # (2, N)
    res.gali[4]      # (2, N)
    res.rem          # (2, N) REM(500), REM(2000)
    res.tune_diffusion, res.qx, res.qy   # (N,)
    res.valid, res.lost_at_turn

``particles`` is not modified. Results are NumPy arrays ordered by the
``particle_id`` of the input particles (``res.particle_id``); orbits lost
before a sample turn give NaN.

Indicators and definitions
==========================

Tangent indicators from ghost particles
---------------------------------------

Each reference particle is tracked together with ``n_ghosts`` *ghost*
particles, displaced by ``displacement`` (``eps``, default ``1e-8``) along
orthogonal directions of a metric (identity, or normalised coordinates for a
lattice). After every ``renorm_every`` turns (and at every sample turn) a
kernel measures the distances ``r_g`` of the ghosts from their reference,
accumulates ``log(r_g / eps)`` and moves the ghosts back to distance ``eps``
along the same directions (no orthogonalisation). Then:

* **FLI(n)** ``= sum_k log(r_k / eps)`` for ghost 0, i.e. the log of the
  stretching of a deviation vector after ``n`` turns; ``res.lyapunov =
  FLI / n`` is the finite-time Lyapunov exponent.
* **Birkhoff FLI** ``= sum_k w_k log(r_k / eps)`` with weights
  ``w_k ∝ exp(-1 / (t (1 - t)))``, ``t = k / (n + 1)``, ``sum w = 1`` (one
  value per sample turn). It converges faster than ``FLI / n``. With
  ``renorm_every > 1`` the chunk weight is the mean of ``w`` over the chunk.
* **SALI** ``= min(|u_0 + u_1|, |u_0 - u_1|)`` of the unit deviation
  vectors of ghosts 0 and 1: ~constant for regular orbits, decays as
  ``exp(-(l1 - l2) n)`` for chaotic ones.
* **GALI_k** (``k = 2 .. n_ghosts``): volume spanned by the first ``k`` unit
  deviation vectors (product of the diagonal of a modified Gram–Schmidt QR).
  On a 2-torus of a 4D map GALI\ :sub:`2` is ~constant, GALI\ :sub:`3` ∝ n\ :sup:`-2`,
  GALI\ :sub:`4` ∝ n\ :sup:`-4`; for chaotic orbits GALI\ :sub:`3` ∝ exp(-2 l1 n) and
  GALI\ :sub:`4` ∝ exp(-4 l1 n).

.. code-block:: python

    res = xt.chaos.compute_tangent_indicators(
        line, particles, num_turns=10_000, sample_turns=[1000, 3000, 10_000],
        n_ghosts=4,          # 1 is enough for FLI only
        displacement=1e-8,   # eps in the metric
        renorm_every=1,      # renormalisation period
        metric=None,         # (4, 4), e.g. xt.chaos.metric_from_twiss(tw)
        alignment=True)      # SALI and GALI

An orbit is *invalid* (``res.valid == False``) as soon as its reference or
any of its ghosts is lost; ``res.lost_at_turn`` is the earliest loss turn.

**Finite-difference floor.** The ghost separation resolves directions to
about ``1e-16 |z| / eps``. ``eps = 1e-8`` is the best compromise between
this round-off floor and the O(eps) nonlinearity error. Consequences:
on chaotic orbits SALI saturates around 1e-7–1e-6 and GALI\ :sub:`4` around
1e-21 (or exactly 0 once two ghosts coincide), so classification thresholds
must be chosen above these floors (e.g. SALI < 1e-5 rather than the usual
1e-8).

Reversibility error (REM)
-------------------------

``REM(n) = |M^-n M^n z - z|`` (in the metric): ``n`` turns forward, then
``n`` turns with ``backtrack=True``. It is driven by round-off: it stays near
1e-16–1e-12 for regular orbits and grows exponentially for chaotic ones.
Compare REM values across machines by orders of magnitude only.

.. code-block:: python

    res = xt.chaos.compute_rem(line, particles, rem_turns=[1000, 3000])
    res.rem    # (2, N)

The line must be backtrackable (every element has ``has_backtrack``; e.g.
``LineSegmentMap`` is not), non-collective and without time-dependent
variables; otherwise a ``ValueError`` names the problem.

Frequency map analysis (FMA)
----------------------------

Tunes in two consecutive windows of ``window`` turns and the tune diffusion
``log10(sqrt((qx1 - qx2)^2 + (qy1 - qy2)^2))``. The default method
accumulates Birkhoff-weighted phase advances in a kernel, with O(1) memory
per particle, through the beam element :class:`xtrack.chaos.BirkhoffTuneMonitor`.
For the best performance, put the monitor in the line, so that all turns run
in a single tracking call (and, with ``compute_indicators``, within the ghost
pass at no extra cost):

.. code-block:: python

    window = 1000
    monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=len(x0.ravel()),
                                           window=window)
    line = xt.Line(elements=[monitor, henon, aperture])   # monitor first
    line.build_tracker()
    res = xt.chaos.compute_fma(line, particles, window=window)

Without a monitor in the line, ``compute_fma`` applies a standalone monitor
between one-turn tracking calls (same results, slower).
``method='naff'`` records turn-by-turn data and uses ``nafflib`` on the
host; it is the reference path.

The Birkhoff tune is a rotation number around the origin of the normalised
plane. It agrees with NAFF when the projected motion winds around the
origin, but it is wrong (with a spurious large diffusion) when a plane is
dominated by other frequencies or offsets, e.g. on the quadratic Hénon map
for ``|x| << y^2``. Use ``method='naff'`` where this matters.

Lattices
========

For a ring, take norms in normalised coordinates and measure tunes around
the closed orbit:

.. code-block:: python

    tw = line.twiss(method='4d')
    metric = xt.chaos.metric_from_twiss(tw)          # W^-1, (x, px, y, py)
    co = [tw.x[0], tw.px[0], tw.y[0], tw.py[0]]
    particles = line.build_particles(method='4d', x_norm=..., y_norm=...,
                                     nemitt_x=2.5e-6, nemitt_y=2.5e-6)
    res = xt.chaos.compute_indicators(line, particles, num_turns=1000,
                                      metric=metric, closed_orbit=co,
                                      displacement=1e-10,
                                      renorm_every=10)

The displacement is in normalised units (sqrt(m)); choose it about 1e-8
times the normalised amplitude. By default (``dim=4``) ghosts are displaced
in ``(x, px, y, py)`` only. With ``dim=6`` they are displaced in
``(x, px, y, py, zeta, pzeta)`` (the order of ``tw.W_matrix``) and norms
include the longitudinal plane; the kernels update ``delta``, ``rpp`` and
``rvv`` of the ghosts consistently with ``pzeta``:

.. code-block:: python

    tw = line.twiss()                                  # 6D
    metric = xt.chaos.metric_from_twiss(tw, dim=6)
    res = xt.chaos.compute_tangent_indicators(
        line, particles, num_turns=1000, n_ghosts=6, dim=6,
        metric=metric, displacement=1e-9)

The Hénon map element
=====================

:class:`xtrack.Henonmap` is a one-turn map: a polynomial kick from normal
multipoles followed by a rotation in normalised phase space, optionally with
twiss normalisation, chromaticity and dispersion. It has an exact inverse
(backtracking). ``multipole_coeffs=[2, 6 mu]`` gives
``px += x^2 - y^2 + mu (x^3 - 3 x y^2)``, ``py += -2 x y + mu (y^3 - 3 x^2 y)``.

Example: stability maps
=======================

.. literalinclude:: ../../xtrack/examples/chaos_indicators/000_henon_stability_maps.py
   :language: python

.. image:: figures/henon_stability_maps.png

Performance
===========

See ``examples/chaos_indicators/001_benchmark.py``. Cost relative to plain
tracking, for the same horizon ``n``:

* ghosts multiply the tracked particles by ``1 + n_ghosts`` (2 for FLI only,
  5 for SALI/GALI with 4 ghosts);
* each renormalisation chunk is a separate tracking call plus two small
  kernels; for cheap one-turn maps this per-call overhead dominates, so use
  ``renorm_every > 1`` where possible (in the tangent-map limit FLI and the
  SALI/GALI directions do not depend on it, the Birkhoff FLI becomes
  chunk-averaged; with ghosts the separation must stay small between
  renormalisations, i.e. ``eps * exp(l1 * renorm_every) << |z|``);
* REM tracks ``2 n`` turns;
* FMA with the in-line monitor adds a negligible per-turn cost on a lattice.

Measured on 4 CPU cores (time relative to plain tracking of the references
for the same number of turns, serial / OpenMP):

========================================  ===============  ==================
Indicator                                 Hénon map        HL-LHC (23.5k el.)
========================================  ===============  ==================
FLI (1 ghost, renormalised every turn)    8.3 / 8.0        1.9 / 1.5
FLI + SALI + GALI (4 ghosts, every turn)  22 / 19          4.5 / 3.1
FLI + SALI + GALI (4 ghosts, every 10)    7.0 / 6.7        4.5 / 3.0
REM(n)                                    2.5 / 2.3        1.9 / 1.8
FMA, Birkhoff, in-line monitor            5.4 / 4.7        0.97 / 0.96
FMA, NAFF on the host                     138 / 283        1.0 / 1.0
All of the above (compute_indicators)     32 / 36          6.1 / 4.7
========================================  ===============  ==================

On a cheap one-turn map the fixed costs dominate (one tracking call and two
small kernels per renormalisation, the atan2 of the tune monitor, NAFF on
the host); on a realistic lattice the cost is set by the number of tracked
particles (``1 + n_ghosts``) and turns (``2 n`` for REM).
