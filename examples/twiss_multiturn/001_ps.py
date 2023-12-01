import numpy as np
from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.input("""
beam, particle=proton, pc = 14.0;
BRHO      = BEAM->PC * 3.3356;
""")
mad.call("ps.seq")
mad.call("ps_hs_sftpro.str")
mad.use('ps')
twm = mad.twiss()

line = xt.Line.from_madx_sequence(mad.sequence.ps, allow_thick=True,
                                  deferred_expressions=True,
                                  )
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                    q0=1, gamma0=mad.sequence.ps.beam.gamma)
line.twiss_default['method'] = '4d'

tw = line.twiss()

opt = line.match(
    solve=False,
    vary=[xt.VaryList(['kf', 'kd'], step=1e-5)],
    targets=[xt.TargetSet(qx=6.255278, qy=6.29826, tol=1e-7)],
)
opt.solve()


r0 = np.linspace(0, 100, 50)
p = line.build_particles(
    x_norm=r0*np.cos(np.pi/20.),
    px_norm=r0*np.sin(np.pi/20.),
    nemitt_x=1e-6, nemitt_y=1e-6)

line.track(p, num_turns=1000, turn_by_turn_monitor=True)
mon = line.record_last_track

tw_mt = line.twiss(co_guess={'x': 0.032}, num_turns=4)
tw_core = line.twiss(co_guess={'x': 0.0}, num_turns=0)

# Inspect and plot
tw_start_turns = tw_mt.rows['_turn_.*']
tw_start_turns.show()
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(mon.x.flatten(), mon.px.flatten(), '.', markersize=1)
plt.plot(tw_start_turns.x, tw_start_turns.px, '*')
plt.ylim(-0.004, 0.004)
plt.xlim(-0.08, 0.08)

plt.figure(2)
ax1 = plt.subplot(3,1,1)
plt.plot(tw_mt.s, tw_mt.x)
plt.plot(tw_mt.s, tw_mt.y)
plt.subplot(3,1,2, sharex=ax1)
plt.plot(tw_mt.s, tw_mt.betx)
plt.plot(tw_core.s, tw_core.betx)
plt.subplot(3,1,3, sharex=ax1)
plt.plot(tw_mt.s, tw_mt.bety)
plt.plot(tw_core.s, tw_core.bety)

plt.show()