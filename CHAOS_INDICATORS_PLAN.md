# Chaos indicators in xtrack: implementation plan for a cloud session

## Context

The review doc "ML for long-term stability in HL-LHC" (tabs *Classifier design* and *Hénon pilot*) builds its
survival classifier on short-run chaos features: FLI with Birkhoff weights, GALI/SALI, REM and tune diffusion,
all computed on GPU for ~10^6 orbits. xtrack has none of this today (no match for lyapunov/FLI/SALI/GALI/FMA
anywhere; `Line.get_footprint` is plain FFT and refuses losses and OpenCL).

Goal: add an **additive, PR-ready** `chaos indicators` feature to xtrack whose per-orbit arithmetic runs in
xobjects kernels (CPU serial, OpenMP, CUDA, OpenCL from one C source), validated on a Hénon map line.

Decisions already taken with the user:

- Start fresh from the papers (Bazzani et al. PRE 107, 064209; Montanari et al. EPJ Plus 140, 2025). No prior code to port.
- Tangent dynamics by **ghost (shadow) particles**; the analytic Hénon tangent map is the *test oracle only*.
- Additive: do not modify `tracker.py` or `Particles`. New module, new elements/kernels, tests, examples.
- Cloud session validates on **`ContextCpu` serial + OpenMP**. GPU code is written to the portable subset; the user runs cupy locally.
- The Hénon element is **not in xtrack** (never merged). It must be ported from
  `https://github.com/deveres99/xtrack/tree/henon-map` (6 commits from 2023, 4062 behind main) and validated as stage 1.

Baseline: upstream `xsuite/xtrack` main @ `c5febfd11` (v0.115.5), xobjects 0.7.1, xpart 0.23.22, xdeps 0.10.21.

## Facts that shape the design (verified in the clones)

| Fact | Where | Consequence |
|---|---|---|
| On GPU `track_line` runs one thread per particle for *all* turns | `xtrack/tracker.py:857-1041` | A reference and its ghost are never synchronised inside one launch. Pair operations must happen **between** launches. |
| Lost particles are swapped on CPU (serial: with last active; OpenMP: per chunk + global reorganise); not on GPU | `particles/headers/local_particle_common.h:291-375` | Never address a partner by slot. Build an id→slot map on device before every pair operation. |
| Killed particles get `x,px,y,py,zeta = 1e30` | `local_particle_common.h:423-431` | An orbit is invalid as soon as the reference *or any* ghost has `state <= 0`; freeze its outputs. |
| `Particles` field list is fixed (X-macros, monitors, prebuilt kernels) | `particles/particles.py:25-82` | No new per-particle fields. Accumulators live in a separate `xo.HybridClass`. `parent_particle_id` is free to mark ghost→reference. |
| No end-of-turn hook; per-turn logic must be a beam element | `tracker.py:910-1041` | Per-particle (partner-free) accumulation, e.g. tunes, is an in-line element. |
| Backtrack is a runtime flag, `XS_FLAG_BACKTRACK`; a line is backtrackable only if every element has `has_backtrack = True` | `tracker.py:1601-1614`, `tracker_data.py:128`, pattern in `elements_src/xyshift.h` | REM uses `line.track(..., backtrack=True)`. The old `#ifdef XSUITE_BACKTRACK` in the Hénon header must be migrated. |
| `LineSegmentMap` is **not** backtrackable; `Multipole`, `Drift`, `Cavity`, `LimitRect`, slices are | `beam_elements/*.py` | The Hénon element needs its own exact inverse. Use `LimitRect` as the loss boundary of the test line. |
| The turn-by-turn `ParticlesMonitor` indexes by `particle_id`, does not log during backtrack | `monitors/particles_monitor.h:12-77` | Fine for the NAFF reference path (forward only). |
| Serial `ContextCpu` refuses to JIT classes that have no prebuilt kernel unless allowed | `xobjects/context_cpu.py:42-80` | Tests use `@allow_kernel_compilation` (`xobjects/test_helpers.py:112`); see how `NonLinearLens`/`Exciter` are listed in `prebuilt_kernel_definitions/element_types.py` and mirror that for new classes. |
| No generic linear algebra in C; `pyopencl.array` has no `linalg` | — | Hand-write fixed-size (≤6×6) modified Gram–Schmidt in the kernel. |
| Model for a HybridClass with its own kernels | `xtrack/multisetter/multisetter.py:7-157` + `multisetter.h` | Template for the pair kernels. |
| Model for a per-particle accumulator element | `monitors/last_turns_monitor.py:34` + `.h:16-55` | Template for the tune monitor. |
| Tests: flat `tests/test_*.py`, `@for_all_test_contexts`, `xo.assert_allclose`, autouse fixture checks no leaked buffers | `tests/conftest.py`, e.g. `tests/test_footprint.py:14-21` | Follow exactly. |

## Architecture

```
xtrack/beam_elements/henonmap.py            Henonmap element (ported)
xtrack/beam_elements/elements_src/henonmap.h
xtrack/chaos/__init__.py                    public API
xtrack/chaos/ghosts.py                      ghost layout + particle construction
xtrack/chaos/tangent.py                     GhostTangent HybridClass (accumulators + kernels)
xtrack/chaos/src/ghost_tangent.h            slot map, renormalise, SALI/GALI, distance kernels
xtrack/chaos/tune_monitor.py                BirkhoffTuneMonitor beam element
xtrack/chaos/src/birkhoff_tune_monitor.h
xtrack/chaos/indicators.py                  drivers: tangent indicators, REM, FMA, combined run
tests/test_henonmap.py
tests/test_chaos_indicators.py
tests/_henon_reference.py                   pure-NumPy oracle (map, analytic Jacobian, indicators)
examples/chaos_indicators/00*_*.py          Hénon stability maps, one per indicator + benchmark
```

Module name `xtrack.chaos` is a proposal; keep whatever reads best next to `xtrack.footprint`.

### 1. Ghost layout (`ghosts.py`)

- One `xt.Particles` object holds `N` references and `N*G` ghosts. Reference `i` has `particle_id = i`;
  ghost `g` of reference `i` has `particle_id = N + i*G + g` and `parent_particle_id = i`.
- Initial displacement of ghost `g`: `eps * e_g`, with `e_g` an orthonormal set in the chosen metric
  (default: canonical basis of the first `G` normalised coordinates). `eps` default `1e-8`.
- Metric: optional 6×6 `W^-1`-like matrix and reference orbit so norms are taken in normalised coordinates
  (identity for the Hénon map; from `tw.W_matrix[0]` / `particle_on_co` for a lattice). `dim = 4` first.
- `dim = 6` is a gated extension (see Stage 7): energy variables must be written consistently
  (`delta`/`pzeta`/`rpp`/`rvv`), so it is not part of the minimum deliverable.

### 2. Pair kernels (`tangent.py`, `ghost_tangent.h`)

`GhostTangent(xo.HybridClass)` with `_depends_on = [xt.Particles]`, fields roughly:
`n_ref, n_ghosts, dim, eps, metric: Float64[36], slot_of_id: Int64[:], valid: Int8[:], lost_at_turn: Int64[:],
log_growth: Float64[:] (N*G), fli_wb: Float64[:] (N*S), sali: Float64[:], gali: Float64[:] (N*(G-1))`.

Kernels (unique, prefixed `c_name`s; `VECTORIZE_OVER`, never the legacy comment form):

1. `build_slot_map(state, particles)`: over slots, `slot_of_id[particle_id[slot]] = slot`. Race-free (ids unique).
2. `renormalise(state, particles, weights*, n_weights, flag_gali)`: over references `i`:
   - locate reference and ghosts through `slot_of_id`; if any has `state <= 0`, clear `valid[i]`, record `lost_at_turn`, return;
   - copy displacements into a local `double d[6][6]`, apply the metric, compute norms `r_g`;
   - `log_growth[i,g] += log(r_g/eps)`; for each sampling horizon `s`, `fli_wb[i,s] += weights[s] * log(r_0/eps)`;
   - if `flag_gali`: unit vectors → SALI = `min(|u0+u1|, |u0-u1|)`; GALI_k = `prod_{j<k} |R_jj|` from modified
     Gram–Schmidt on the unit vectors (avoids the squared conditioning of a Gram determinant);
   - write each ghost back at `ref + eps * d_g / r_g` (direction preserved, **no orthogonalisation**: SALI/GALI need it that way).
   - Thread `i` writes only its own ghosts and its own accumulator rows.
3. `distance_to(state, particles, x0*, px0*, y0*, py0*, out*)`: per `particle_id`, metric distance to stored coordinates (REM).

Driver loop (Python, on-device data only, no host transfer):
`for chunk in schedule: line.track(p, num_turns=chunk); build_slot_map; renormalise`.
The schedule is every `renorm_every` turns (default 1 for maps, user-settable for lattices) plus all `sample_turns`.
At each sample turn the accumulators are snapshotted into the result.

One ghost set serves all tangent indicators at once: FLI from ghost 0, SALI from ghosts 0–1, GALI_k from ghosts 0..k-1.

### 3. Indicator definitions (state them in docstrings exactly)

- **FLI(n)** = `log_growth[i,0]` = Σ log(r_k/eps); finite-time Lyapunov estimate = FLI/n.
- **FLI with Birkhoff weights** = Σ_k w(k/n) log(r_k/eps), `w(t) ∝ exp(-1/(t(1-t)))`, Σw = 1. Weights depend on
  the horizon `n`, so one accumulator per sample turn; weights are computed on host and passed per chunk.
  For `renorm_every > 1` the chunk weight is the sum over its turns (document as an approximation; exact at 1).
- **SALI**, **GALI_k** (k = 2..G) as above.
- **REM(n)**: copy particles on device, track `n` forward, `n` with `backtrack=True`, metric distance to the start.
  Multiple horizons = one forward pass with `particles.copy()` at each checkpoint, backtrack each copy.
  Raise a clear error if the line is not backtrackable (`line.tracker._tracker_data_base._is_backtrackable`),
  collective, or has time-dependent vars.
- **FMA**: tune diffusion `log10 sqrt(Δqx² + Δqy²)` between two windows.
  - *In-kernel path (default)*: `BirkhoffTuneMonitor`, an in-line element (`behaves_like_drift`, `has_backtrack`
    returning a marker like the monitors). Per particle, indexed by `particle_id`: normalised phase
    `θ = atan2(-px_n, x_n)` per plane, unwrapped advance since the previous turn, accumulated with Birkhoff
    weights over two fixed windows. O(1) memory per particle, all contexts.
  - *Reference path*: `ParticlesMonitor` + `nafflib` on host (optional import, as `line.py:3454` does). Used for validation.

### 4. Public API (proposal, refine while implementing)

```python
res = xt.chaos.compute_indicators(line, particles, num_turns=10_000, sample_turns=(1000, 3000, 10_000),
                                  indicators=('fli', 'sali', 'gali', 'rem', 'fma'),
                                  n_ghosts=4, displacement=1e-8, renorm_every=1, metric=None)
res.fli, res.fli_birkhoff, res.sali, res.gali[k], res.rem, res.qx, res.qy, res.tune_diffusion, res.valid, res.lost_at_turn
```

Plus the single-purpose functions it is built from. Results are host NumPy arrays ordered by reference `particle_id`.
Do **not** add `Line` methods in this branch; that is a one-line follow-up once the API settles.

## Stages (commit and push after each; run the stage's tests on both contexts first)

**Stage 0 — environment and baseline.** Dev-install (see *Deploying*). Run `tests/test_monitor.py`,
`tests/test_footprint.py`, `tests/test_tracker.py::test_backtrack_with_flag` on `ContextCpu;ContextCpu:auto`.
If the baseline fails, stop and report; do not work around it.

**Stage 1 — port and validate `Henonmap`.** Source: branch `henon-map` of `deveres99/xtrack`
(files `xtrack/beam_elements/elements_src/henonmap.h`, the `Henonmap` class added to the now-removed
`beam_elements/elements.py`, `tests/test_henonmap.py`). Cherry-pick will not apply; port by hand and keep
authorship with `Co-authored-by:` trailers taken from the original commits.
- New module `beam_elements/henonmap.py`, exported in `beam_elements/__init__.py`; `allow_rot_and_shift = False`.
- Header: `#include "xtrack/headers/track.h"`, `GPUFUN`, `START_PER_PARTICLE_BLOCK`/`END_PER_PARTICLE_BLOCK`;
  replace `#ifdef XSUITE_BACKTRACK` by `LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)`.
- The header copies coefficients into `[128]` local arrays with no bound check: validate lengths in `__init__`.
- Check the kick convention against the pilot map: `multipole_coeffs=[2, 6μ]` must give
  `px + x² − y² + μ(x³ − 3xy²)`, `py − 2xy + μ(y³ − 3x²y)`, followed by the rotation.
- Tests: the original test updated (`xt.Particles`), one-turn analytic formula, forward+backtrack round trip
  (1e-12), symplecticity of the finite-difference Jacobian, `norm=False` vs `norm=True` equivalence,
  `_henon_reference.py` vs xtrack over 10^3 turns, both contexts.
- Out of scope: tune modulation (needs per-turn data; note as follow-up for the pilot).

**Stage 2 — NumPy oracle** (`tests/_henon_reference.py`): map, exact Jacobian, tangent-vector propagation with
the *same* renormalisation schedule, reference FLI / Birkhoff FLI / SALI / GALI / REM / Birkhoff tune.

**Stage 3 — ghosts, pair kernels, FLI.** Tests: ghost vs analytic-tangent FLI on regular orbits (state the
tolerance found and its dependence on `eps`; expect error O(eps)); serial vs OpenMP agreement;
loss handling with a grid straddling the stability border (lost orbits flagged, `lost_at_turn` right,
no NaN in survivors, results invariant under particle reordering).

**Stage 4 — SALI and GALI.** Tests: regular orbit → SALI and GALI_2 ~ constant, GALI_3 ∝ n⁻², GALI_4 ∝ n⁻⁴
(fit the slope above the finite-difference floor ~1e-16/eps and document the floor);
chaotic orbit → exponential decay, slope consistent with the Lyapunov estimate from FLI.

**Stage 5 — REM.** Tests: regular orbit REM(10³) near round-off; chaotic orbit saturates at O(1);
non-backtrackable line (e.g. with `LineSegmentMap`) raises. REM is round-off driven, so compare contexts by
classification, not by value.

**Stage 6 — FMA.** `BirkhoffTuneMonitor` + NAFF reference path. Tests: Birkhoff tune vs `nafflib` on regular
orbits; small-amplitude tune → linear tune; diffusion separates regular from chaotic orbits.

**Stage 7 — integration.** `compute_indicators`, examples reproducing a Hénon stability map per indicator
(grid in (x, y), ω = 2π·(0.168, 0.201), PNGs attached to the report), a throughput benchmark
(particle-turns/s, plain tracking vs each indicator, both contexts), a docs draft `chaos_indicators.rst`
in the style of `xsuite/docs/footprint.rst` (the docs repo is separate; leave the draft in the PR description or `examples/`).
Only if everything above is green: `dim = 6` on a small FODO + cavity line.

## Kernel portability rules (GPU is not tested in the cloud, so follow these strictly)

- Common subset of C99 / CUDA C++ / OpenCL C 2.0: no VLAs, compound literals, designated initialisers, recursion,
  function pointers, `malloc`, `printf`; fixed-size local arrays only.
- `GPUFUN`, `GPUKERN`, `GPUGLMEM`, `RESTRICT` macros; `VECTORIZE_OVER(i, n) … END_VECTORIZE`; guard `<math.h>`
  with `#ifdef XO_CONTEXT_CPU`; `PI` from `xtrack/headers/constants.h`; `intNN_t` types only.
- Kernels are `void`; outputs go to arrays. Scalar `xo.Arg` types must match the C signature exactly.
- Pointer args are nplike arrays, never xobject Arrays; struct data is reached through generated accessors.
- Each iteration writes only to data owned by its index. Copy device data into private locals, compute, write back.
- No empty structs; include guards on every header.

## Verification

```bash
export XOBJECTS_TEST_CONTEXTS="ContextCpu;ContextCpu:auto" XSUITE_ALLOW_KERNEL_COMPILATION=1
python -m pytest -v tests/test_henonmap.py tests/test_chaos_indicators.py
python -m pytest -v tests/test_monitor.py tests/test_footprint.py tests/test_tracker.py tests/test_elements.py   # no regressions
python examples/chaos_indicators/000_henon_stability_maps.py                                                    # figures + benchmark
```

Then once more **without** `XSUITE_ALLOW_KERNEL_COMPILATION` to prove the `@allow_kernel_compilation`
decorators and class attributes are sufficient. New tests must total under ~2 minutes per context.

Final report (PR description on the fork): what was built, tolerances actually achieved per indicator,
the finite-difference floor, benchmark table, figures, every deviation from this plan, and the explicit
statement that CUDA/OpenCL were not executed. Local follow-up for the user:
`XOBJECTS_TEST_CONTEXTS="ContextCupy" pytest -m context_dependent tests/test_henonmap.py tests/test_chaos_indicators.py`.

Stop and report instead of improvising if: the baseline fails, an indicator cannot meet its oracle check,
or a change to `tracker.py` / `Particles` looks unavoidable.

## Environment and working rules for the executing session

- You are on branch `feature/chaos-indicators` of a fork of `xsuite/xtrack`, created from upstream main @ `c5febfd11`.
  The Hénon source is mirrored in the same fork as branch `upstream-henon-map`
  (`git fetch origin upstream-henon-map`); if it is missing, try
  `git fetch https://github.com/deveres99/xtrack henon-map`, and if that also fails, stop and report.
- If the packages are not installed yet (needs PyPI and a C compiler with OpenMP):
  ```bash
  pip install "xobjects==0.7.1" "xdeps==0.10.21" "xpart==0.23.22" pytest pytest-mock nafflib matplotlib
  pip install pandas scipy tqdm requests "madng-tpsa>=0.3.1" && pip install -e . --no-deps
  ```
  Do not install `xsuite` or the full `[tests]` extra (cpymad, PyHEADTAIL, pymadng are not needed).
  This script was written without being run; fix it as needed and record the working version in the report.
- One commit per stage, pushed, with that stage's tests passing on both CPU contexts.
- Do not open a pull request against `xsuite/xtrack`. Do not modify `tracker.py` or `Particles`.
- Leave this file in place; the user removes it before any upstream PR.
