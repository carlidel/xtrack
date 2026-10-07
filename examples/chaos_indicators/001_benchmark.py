"""Cost of the chaos indicators compared with plain tracking.

Two regimes:
- Henon map: one very cheap element per turn (~18 ns per particle-turn),
  so the per-orbit work of the pair kernels and of the tune monitor and the
  Python cost of one tracking call per renormalisation chunk are exposed;
- HL-LHC (test_data/hllhc15_noerrors_nobb): ~23500 elements per turn,
  tracking dominates.

Every row computes its indicator at the same horizon n (``num_turns``):
REM(n) tracks n turns forward and n backward, FMA uses two windows of
(n - 1) // 2 turns. Times exclude kernel compilation (a warm-up call is
made first). Usage: python 001_benchmark.py [henon|hllhc|all]
"""

import sys
import time

import numpy as np

import xobjects as xo
import xtrack as xt

# The HL-LHC kernel is compiled here (no prebuilt kernels needed)
xo.settings.allow_kernel_compilation = True

CONTEXTS = {'ContextCpu (serial)': lambda: xo.ContextCpu(),
            'ContextCpu (OpenMP, auto)': lambda: xo.ContextCpu(omp_num_threads='auto')}


def timed(fn, repeat=1):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_cases(line, line_no_monitor, make_particles, n_ref, num_turns,
              metric=None, closed_orbit=None, displacement=1e-8,
              include_standalone=True):
    window = (num_turns - 1) // 2
    ci = xt.chaos

    cases = {
        'plain tracking, 1 call': lambda: line_no_monitor.track(
            make_particles(), num_turns=num_turns),
        'plain tracking, 1 call per turn': lambda: [
            line_no_monitor.track(p, num_turns=1)
            for p in [make_particles()] for _ in range(num_turns)],
        'FLI (1 ghost, renorm every turn)': lambda: ci.compute_tangent_indicators(
            line_no_monitor, make_particles(), num_turns, n_ghosts=1,
            metric=metric, displacement=displacement),
        'FLI+SALI+GALI (4 ghosts, renorm every turn)': lambda: ci.compute_tangent_indicators(
            line_no_monitor, make_particles(), num_turns, n_ghosts=4,
            metric=metric, displacement=displacement),
        'FLI+SALI+GALI (4 ghosts, renorm every 10 turns)': lambda: ci.compute_tangent_indicators(
            line_no_monitor, make_particles(), num_turns, n_ghosts=4,
            renorm_every=10, metric=metric, displacement=displacement),
        'REM(n) (n forward + n backward)': lambda: ci.compute_rem(
            line_no_monitor, make_particles(), num_turns, metric=metric),
        'FMA Birkhoff, in-line monitor': lambda: ci.compute_fma(
            line, make_particles(), window, metric=metric,
            closed_orbit=closed_orbit),
    }
    if include_standalone:
        cases['FMA Birkhoff, standalone monitor'] = lambda: ci.compute_fma(
            line_no_monitor, make_particles(), window, metric=metric,
            closed_orbit=closed_orbit)
    cases['FMA NAFF (turn-by-turn + nafflib on host)'] = lambda: ci.compute_fma(
        line_no_monitor, make_particles(), window, metric=metric,
        closed_orbit=closed_orbit, method='naff')
    cases['all (compute_indicators, in-line FMA)'] = lambda: ci.compute_indicators(
        line, make_particles(), num_turns, n_ghosts=4, metric=metric,
        closed_orbit=closed_orbit, displacement=displacement,
        fma_window=window)

    out = {}
    for name, fn in cases.items():
        fn()  # warm-up (compilation)
        out[name] = timed(fn)
        print(f'    {name:50s} {out[name]:8.3f} s', flush=True)
    return out


def henon_benchmark(context, n_ref=10_000, num_turns=1000):
    window = (num_turns - 1) // 2
    henon = xt.Henonmap(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201,
                        multipole_coeffs=[2.0], norm=True)
    aperture = xt.LimitRect(min_x=-1, max_x=1, min_y=-1, max_y=1)
    monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=n_ref, window=window)
    line = xt.Line(elements=[monitor, henon.copy(), aperture.copy()])
    line.build_tracker(_context=context)
    line_nm = xt.Line(elements=[henon, aperture])
    line_nm.build_tracker(_context=context)

    rng = np.random.default_rng(0)
    x0, y0 = rng.uniform(0, 0.3, size=(2, n_ref))  # mostly surviving orbits

    def make_particles():
        return xt.Particles(x=x0, y=y0, _context=context)

    return run_cases(line, line_nm, make_particles, n_ref, num_turns)


def hllhc_benchmark(context, n_ref=200, num_turns=30):
    window = (num_turns - 1) // 2
    fname = '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json'
    line_nm = xt.load(fname)
    line_nm.set_particle_ref('proton', p0c=7e12)
    line_nm.build_tracker(_context=context)
    tw = line_nm.twiss(method='4d')
    metric = xt.chaos.metric_from_twiss(tw)
    closed_orbit = np.array([tw.x[0], tw.px[0], tw.y[0], tw.py[0]])

    line = xt.load(fname)
    line.set_particle_ref('proton', p0c=7e12)
    line.insert('tune_monitor', xt.chaos.BirkhoffTuneMonitor(
        num_particles=n_ref, window=window, metric=metric,
        closed_orbit=closed_orbit), at=0.0)
    line.build_tracker(_context=context)

    amp = np.linspace(2, 12, n_ref)

    def make_particles():
        return line_nm.build_particles(
            method='4d', x_norm=amp / np.sqrt(2), y_norm=amp / np.sqrt(2),
            nemitt_x=2.5e-6, nemitt_y=2.5e-6, delta=0)

    return run_cases(line, line_nm, make_particles, n_ref, num_turns,
                     metric=metric, closed_orbit=closed_orbit,
                     displacement=1e-10, include_standalone=False)


def table(results, n_ref, num_turns, title):
    names = list(next(iter(results.values())).keys())
    ctx_names = list(results.keys())
    lines = [f'### {title}', '',
             f'{n_ref} reference orbits, horizon n = {num_turns} turns. '
             'Time in s; x = time / plain tracking (1 call); '
             'orbit-turns/s = references x n / time.', '',
             '| Indicator | ' + ' | '.join(
                 f'{cc}: time | x | orbit-turns/s' for cc in ctx_names) + ' |',
             '|---|' + '---:|---:|---:|' * len(ctx_names)]
    for nn in names:
        row = [nn]
        for cc in ctx_names:
            tt = results[cc][nn]
            t0 = results[cc]['plain tracking, 1 call']
            row += [f'{tt:.3g}', f'{tt / t0:.2f}', f'{n_ref * num_turns / tt:.3g}']
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    report = []
    if which in ('henon', 'all'):
        res = {}
        for cname, mk in CONTEXTS.items():
            print(f'Henon map, {cname}')
            res[cname] = henon_benchmark(mk())
        report.append(table(res, 10_000, 1000, 'Henon map'))
    if which in ('hllhc', 'all'):
        res = {}
        for cname, mk in CONTEXTS.items():
            print(f'HL-LHC, {cname}')
            res[cname] = hllhc_benchmark(mk())
        report.append(table(res, 200, 30, 'HL-LHC v1.5 (23.5k elements)'))
    text = '\n\n'.join(report)
    print(text)
    with open(f'chaos_benchmark_{which}.md', 'w') as fid:
        fid.write(text + '\n')
