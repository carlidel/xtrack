"""Stability maps of the 4D Henon map with all the chaos indicators.

Grid of initial conditions (x, y), px = py = 0, in normalised coordinates,
omega = 2 pi (0.168, 0.201), quadratic kick (mu = 0). The short-run
indicators (2000 turns) are compared with the survival time over 1e5 turns.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import xobjects as xo
import xtrack as xt

context = xo.ContextCpu(omp_num_threads='auto')

n_grid = 100
x_max = 0.5
num_turns = 2000
survival_turns = 100_000
mu = 0.0

# Line: Henon map, tune monitor at the start (in-line FMA), loss boundary
fma_window = (num_turns - 1) // 2
xx, yy = np.meshgrid(np.linspace(x_max / n_grid, x_max, n_grid),
                     np.linspace(x_max / n_grid, x_max, n_grid))
x0, y0 = xx.ravel(), yy.ravel()
n_part = len(x0)

henon = xt.Henonmap(omega_x=2 * np.pi * 0.168, omega_y=2 * np.pi * 0.201,
                    multipole_coeffs=[2.0, 6.0 * mu], norm=True)
tune_monitor = xt.chaos.BirkhoffTuneMonitor(num_particles=n_part,
                                            window=fma_window)
aperture = xt.LimitRect(min_x=-1, max_x=1, min_y=-1, max_y=1)
line = xt.Line(elements=[tune_monitor, henon, aperture],
               element_names=['tune_monitor', 'henon', 'aperture'])
line.build_tracker(_context=context)

particles = xt.Particles(x=x0, y=y0, _context=context)

t0 = time.perf_counter()
res = xt.chaos.compute_indicators(
    line, particles, num_turns=num_turns, sample_turns=[500, num_turns],
    indicators=('fli', 'sali', 'gali', 'rem', 'fma'), n_ghosts=4,
    displacement=1e-8, fma_window=fma_window)
print(f'Indicators for {n_part} orbits, {num_turns} turns: '
      f'{time.perf_counter() - t0:.1f} s ({res.fma_method})')

# Survival time over many turns (plain tracking)
p_surv = xt.Particles(x=x0, y=y0, _context=context)
t0 = time.perf_counter()
line.track(p_surv, num_turns=survival_turns)
print(f'Survival over {survival_turns} turns: {time.perf_counter() - t0:.1f} s')
p_surv.sort(interleave_lost_particles=True)
survival = np.where(p_surv.state > 0, survival_turns, p_surv.at_turn)

# Reference FMA with NAFF on turn-by-turn data (host, slower)
line_naff = xt.Line(elements=[henon.copy(), aperture.copy()])
line_naff.build_tracker(_context=context)
t0 = time.perf_counter()
res_naff = xt.chaos.compute_fma(line_naff, xt.Particles(x=x0, y=y0, _context=context),
                                window=fma_window, method='naff')
print(f'FMA with NAFF: {time.perf_counter() - t0:.1f} s')

# Plots
def show(ax, values, title, log=False, cmap='viridis', vmin=None, vmax=None):
    vv = np.array(values, dtype=float)
    if log:
        with np.errstate(divide='ignore', invalid='ignore'):
            vv = np.log10(vv)
    im = ax.pcolormesh(xx, yy, vv.reshape(xx.shape), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

plt.close('all')
fig, axs = plt.subplots(2, 4, figsize=(18, 8.5))
show(axs[0, 0], survival, f'log10 survival turns (max {survival_turns})', log=True)
show(axs[0, 1], res.fli[-1] / num_turns, f'FLI / n, n = {num_turns}',
     cmap='magma')
show(axs[0, 2], res.fli_birkhoff[-1], 'Birkhoff-weighted FLI', cmap='magma')
show(axs[0, 3], res.sali[-1], 'log10 SALI', log=True, cmap='magma_r')
show(axs[1, 0], res.gali[4][-1], 'log10 GALI_4', log=True, cmap='magma_r')
show(axs[1, 1], res.rem[-1], f'log10 REM({num_turns})', log=True, cmap='magma')
show(axs[1, 2], res.tune_diffusion, 'log10 tune diffusion', cmap='magma',
     vmin=-12, vmax=-1)
ax = axs[1, 3]
ok = res.fma_valid
sc = ax.scatter(res.qx[ok], res.qy[ok], c=res.tune_diffusion[ok], s=2,
                cmap='magma', vmin=-12, vmax=-1)
ax.set_xlabel('qx')
ax.set_ylabel('qy')
ax.set_title('Frequency map')
plt.colorbar(sc, ax=ax)
fig.suptitle(f'Henon map, omega = 2 pi (0.168, 0.201), mu = {mu}')
fig.tight_layout()
fig.savefig('henon_stability_maps.png', dpi=110)

# How well do the short-run indicators predict long-term survival?
stable = survival == survival_turns
valid = res.valid[-1]
print(f'Lost within {num_turns} turns: {np.sum(~valid)}; '
      f'lost within {survival_turns} turns: {np.sum(~stable)}')
for name, values, larger_is_chaotic in (
        ('FLI/n', res.fli[-1] / num_turns, True),
        ('SALI', res.sali[-1], False),
        ('GALI_4', res.gali[4][-1], False),
        ('REM', res.rem[-1], True),
        ('tune diffusion', res.tune_diffusion, True),
        ('NAFF diffusion', res_naff.tune_diffusion, True)):
    vv = values[valid & np.isfinite(values)]
    lab = stable[valid & np.isfinite(values)]
    # Fraction of correctly ordered (stable, unstable) pairs (ROC AUC)
    score = vv if larger_is_chaotic else -vv
    s_st, s_un = score[lab], score[~lab]
    if len(s_un) == 0 or len(s_st) == 0:
        continue
    order = np.argsort(np.concatenate([s_st, s_un]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[len(s_st):].sum() - len(s_un) * (len(s_un) + 1) / 2) / (
        len(s_st) * len(s_un))
    print(f'  {name:15s} AUC(survivors at {num_turns} turns, lost before '
          f'{survival_turns}) = {auc:.3f}')

plt.show()
