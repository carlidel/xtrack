// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

// Tunes from Birkhoff-weighted averages of the phase advance, accumulated
// in-line, per particle, with O(1) memory. The particle with particle_id
// particle_id_start + i owns entry i of every per-particle array. At turn
// start_turn + j (j = 0..2 window) the monitor measures the normalised
// phases theta = atan2(-p_n, q_n) per plane; the advance of observation j
// (j >= 1), mod 2 pi, is accumulated with weight weights[j - 1] in window 1
// (j <= window) and weights[j - window - 1] in window 2.

#ifndef XTRACK_BIRKHOFF_TUNE_MONITOR_H
#define XTRACK_BIRKHOFF_TUNE_MONITOR_H

#ifdef XO_CONTEXT_CPU
#include <math.h>
#endif  // XO_CONTEXT_CPU

#include "xtrack/headers/track.h"
#include "xtrack/headers/constants.h"


GPUFUN
double BirkhoffTuneMonitor_advance(double theta, double theta_prev)
{
    double adv = theta - theta_prev;
    adv -= 2 * PI * floor(adv / (2 * PI));  // in [0, 2 pi)
    return adv;
}


GPUFUN
void BirkhoffTuneMonitor_track_local_particle(
    BirkhoffTuneMonitorData el, LocalParticle* part0)
{
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        return; // Passive in backtracking
    }

    int64_t const start_turn = BirkhoffTuneMonitorData_get_start_turn(el);
    int64_t const window = BirkhoffTuneMonitorData_get_window(el);
    int64_t const pid_start = BirkhoffTuneMonitorData_get_particle_id_start(el);
    int64_t const num_particles = BirkhoffTuneMonitorData_get_num_particles(el);

    double w_inv[16];
    for (int64_t kk = 0; kk < 16; kk++){
        w_inv[kk] = BirkhoffTuneMonitorData_get_w_inv(el, kk);
    }
    double co[4];
    for (int64_t kk = 0; kk < 4; kk++){
        co[kk] = BirkhoffTuneMonitorData_get_closed_orbit(el, kk);
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        int64_t const ip = LocalParticle_get_particle_id(part) - pid_start;
        int64_t const jj = LocalParticle_get_at_turn(part) - start_turn;

        if (ip >= 0 && ip < num_particles && jj >= 0 && jj <= 2 * window){
            double zz[4];
            zz[0] = LocalParticle_get_x(part) - co[0];
            zz[1] = LocalParticle_get_px(part) - co[1];
            zz[2] = LocalParticle_get_y(part) - co[2];
            zz[3] = LocalParticle_get_py(part) - co[3];
            double zn[4];
            for (int64_t kk = 0; kk < 4; kk++){
                zn[kk] = 0;
                for (int64_t ll = 0; ll < 4; ll++){
                    zn[kk] += w_inv[4 * kk + ll] * zz[ll];
                }
            }
            double const theta_x = atan2(-zn[1], zn[0]);
            double const theta_y = atan2(-zn[3], zn[2]);

            if (jj >= 1){
                double const adv_x = BirkhoffTuneMonitor_advance(
                    theta_x, BirkhoffTuneMonitorData_get_phase_x(el, ip));
                double const adv_y = BirkhoffTuneMonitor_advance(
                    theta_y, BirkhoffTuneMonitorData_get_phase_y(el, ip));
                if (jj <= window){
                    double const ww = BirkhoffTuneMonitorData_get_weights(el, jj - 1)
                                      / (2 * PI);
                    BirkhoffTuneMonitorData_set_qx1(el, ip,
                        BirkhoffTuneMonitorData_get_qx1(el, ip) + ww * adv_x);
                    BirkhoffTuneMonitorData_set_qy1(el, ip,
                        BirkhoffTuneMonitorData_get_qy1(el, ip) + ww * adv_y);
                }
                else{
                    double const ww = BirkhoffTuneMonitorData_get_weights(
                                          el, jj - window - 1) / (2 * PI);
                    BirkhoffTuneMonitorData_set_qx2(el, ip,
                        BirkhoffTuneMonitorData_get_qx2(el, ip) + ww * adv_x);
                    BirkhoffTuneMonitorData_set_qy2(el, ip,
                        BirkhoffTuneMonitorData_get_qy2(el, ip) + ww * adv_y);
                }
                BirkhoffTuneMonitorData_set_n_advances(el, ip,
                    BirkhoffTuneMonitorData_get_n_advances(el, ip) + 1);
            }
            BirkhoffTuneMonitorData_set_phase_x(el, ip, theta_x);
            BirkhoffTuneMonitorData_set_phase_y(el, ip, theta_y);
        }
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_BIRKHOFF_TUNE_MONITOR_H */
