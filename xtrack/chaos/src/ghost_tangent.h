// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

// Pair kernels for tangent-map chaos indicators computed with ghost
// (shadow) particles. Every reference orbit i owns n_ghosts ghosts; a
// reference has particle_id = i, its ghost g has particle_id
// n_ref + i * n_ghosts + g. Partners are located through slot_of_id, which
// build_slot_map must refresh after every tracking call (particles may be
// reordered on CPU). Each thread writes only data owned by its index.

#ifndef XTRACK_GHOST_TANGENT_H
#define XTRACK_GHOST_TANGENT_H

#ifdef XO_CONTEXT_CPU
#include <math.h>
#endif  // XO_CONTEXT_CPU

#include "xobjects/headers/common.h"

#define GHOST_TANGENT_MAX_DIM 6


// Phase-space coordinate k of a slot: x, px, y, py, zeta, pzeta
GPUFUN
double GhostTangent_get_coord(ParticlesData particles, int64_t slot, int64_t k)
{
    if (k == 0) return ParticlesData_get_x(particles, slot);
    if (k == 1) return ParticlesData_get_px(particles, slot);
    if (k == 2) return ParticlesData_get_y(particles, slot);
    if (k == 3) return ParticlesData_get_py(particles, slot);
    if (k == 4) return ParticlesData_get_zeta(particles, slot);
    return ParticlesData_get_pzeta(particles, slot);
}


// Write coordinate k; pzeta updates delta, rpp and rvv consistently (as
// LocalParticle_update_pzeta)
GPUFUN
void GhostTangent_set_coord(ParticlesData particles, int64_t slot, int64_t k,
                            double value)
{
    if (k == 0) ParticlesData_set_x(particles, slot, value);
    else if (k == 1) ParticlesData_set_px(particles, slot, value);
    else if (k == 2) ParticlesData_set_y(particles, slot, value);
    else if (k == 3) ParticlesData_set_py(particles, slot, value);
    else if (k == 4) ParticlesData_set_zeta(particles, slot, value);
    else {
        double const beta0 = ParticlesData_get_beta0(particles, slot);
        double const ptau = value * beta0;
        double const irpp = sqrt(ptau * ptau + 2.0 * value + 1.0);
        ParticlesData_set_pzeta(particles, slot, value);
        ParticlesData_set_delta(particles, slot, irpp - 1.0);
        ParticlesData_set_rpp(particles, slot, 1.0 / irpp);
        ParticlesData_set_rvv(particles, slot, irpp / (1.0 + beta0 * ptau));
    }
}


// Euclidean norm of metric * vec (metric is dim x dim, row major)
GPUFUN
double GhostTangent_apply_metric(GhostTangentData state, int64_t dim,
                                 const double* vec, double* out)
{
    double norm2 = 0;
    for (int64_t kk = 0; kk < dim; kk++){
        double acc = 0;
        for (int64_t ll = 0; ll < dim; ll++){
            acc += GhostTangentData_get_metric(state, kk * dim + ll) * vec[ll];
        }
        out[kk] = acc;
        norm2 += acc * acc;
    }
    return sqrt(norm2);
}


GPUKERN
void GhostTangent_build_slot_map(
    GhostTangentData state,
    ParticlesData particles,
    int64_t n_slots)
{
    int64_t const n_ids = GhostTangentData_len_slot_of_id(state);

    VECTORIZE_OVER(slot, n_slots);
        int64_t const pid = ParticlesData_get_particle_id(particles, slot);
        if (pid >= 0 && pid < n_ids){
            GhostTangentData_set_slot_of_id(state, pid, slot);
        }
    END_VECTORIZE;
}


GPUKERN
void GhostTangent_renormalise(
    GhostTangentData state,
    ParticlesData particles,
    GPUGLMEM double* weights,  // [n_chunks, n_samples], row major
    int64_t chunk_index,
    int64_t flag_gali,
    int64_t n_ref)
{
    int64_t const n_ghosts = GhostTangentData_get_n_ghosts(state);
    int64_t const dim = GhostTangentData_get_dim(state);
    int64_t const n_samples = GhostTangentData_get_n_samples(state);
    double const eps = GhostTangentData_get_eps(state);

    VECTORIZE_OVER(ii, n_ref);

    if (GhostTangentData_get_valid(state, ii)){

        int64_t const slot_ref = GhostTangentData_get_slot_of_id(state, ii);

        // An orbit is invalid as soon as its reference or any ghost is lost
        int64_t lost_turn = -1;
        if (ParticlesData_get_state(particles, slot_ref) <= 0){
            lost_turn = ParticlesData_get_at_turn(particles, slot_ref);
        }
        for (int64_t gg = 0; gg < n_ghosts; gg++){
            int64_t const slot_g = GhostTangentData_get_slot_of_id(
                state, n_ref + ii * n_ghosts + gg);
            if (ParticlesData_get_state(particles, slot_g) <= 0){
                int64_t const tt = ParticlesData_get_at_turn(particles, slot_g);
                if (lost_turn < 0 || tt < lost_turn) lost_turn = tt;
            }
        }

        double ref[GHOST_TANGENT_MAX_DIM];
        double dphys[GHOST_TANGENT_MAX_DIM][GHOST_TANGENT_MAX_DIM];
        double unit[GHOST_TANGENT_MAX_DIM][GHOST_TANGENT_MAX_DIM];
        double rr[GHOST_TANGENT_MAX_DIM];
        for (int64_t gg = 0; gg < GHOST_TANGENT_MAX_DIM; gg++){
            rr[gg] = 1.0;
        }

        if (lost_turn < 0){
            for (int64_t kk = 0; kk < dim; kk++){
                ref[kk] = GhostTangent_get_coord(particles, slot_ref, kk);
            }
            for (int64_t gg = 0; gg < n_ghosts; gg++){
                int64_t const slot_g = GhostTangentData_get_slot_of_id(
                    state, n_ref + ii * n_ghosts + gg);
                for (int64_t kk = 0; kk < dim; kk++){
                    dphys[gg][kk] = GhostTangent_get_coord(particles, slot_g, kk)
                                    - ref[kk];
                }
                rr[gg] = GhostTangent_apply_metric(state, dim, dphys[gg], unit[gg]);
                // Also catches NaN and overflow
                if (!(rr[gg] > 0.0 && rr[gg] < 1e300)){
                    lost_turn = ParticlesData_get_at_turn(particles, slot_ref);
                }
            }
        }

        if (lost_turn >= 0){
            GhostTangentData_set_valid(state, ii, 0);
            GhostTangentData_set_lost_at_turn(state, ii, lost_turn);
        }
        else{
            // Stretching
            for (int64_t gg = 0; gg < n_ghosts; gg++){
                double const lg = log(rr[gg] / eps);
                int64_t const idx = ii * n_ghosts + gg;
                GhostTangentData_set_log_growth(state, idx,
                    GhostTangentData_get_log_growth(state, idx) + lg);
                for (int64_t kk = 0; kk < dim; kk++){
                    unit[gg][kk] /= rr[gg];
                }
            }
            double const lg0 = log(rr[0] / eps);
            for (int64_t ss = 0; ss < n_samples; ss++){
                double const ww = weights[chunk_index * n_samples + ss];
                int64_t const idx = ii * n_samples + ss;
                GhostTangentData_set_fli_wb(state, idx,
                    GhostTangentData_get_fli_wb(state, idx) + ww * lg0);
            }

            // Alignment indices from the unit deviation vectors
            if (flag_gali && n_ghosts >= 2){
                double sp = 0;
                double sm = 0;
                for (int64_t kk = 0; kk < dim; kk++){
                    sp += (unit[0][kk] + unit[1][kk]) * (unit[0][kk] + unit[1][kk]);
                    sm += (unit[0][kk] - unit[1][kk]) * (unit[0][kk] - unit[1][kk]);
                }
                GhostTangentData_set_sali(state, ii, sqrt(sp < sm ? sp : sm));

                // GALI_k = prod_{j < k} |R_jj|, modified Gram-Schmidt
                double qq[GHOST_TANGENT_MAX_DIM][GHOST_TANGENT_MAX_DIM];
                for (int64_t gg = 0; gg < n_ghosts; gg++){
                    for (int64_t kk = 0; kk < dim; kk++){
                        qq[gg][kk] = unit[gg][kk];
                    }
                }
                double gali = 1.0;
                for (int64_t jj = 0; jj < n_ghosts; jj++){
                    for (int64_t pp = 0; pp < jj; pp++){
                        double proj = 0;
                        for (int64_t kk = 0; kk < dim; kk++){
                            proj += qq[pp][kk] * qq[jj][kk];
                        }
                        for (int64_t kk = 0; kk < dim; kk++){
                            qq[jj][kk] -= proj * qq[pp][kk];
                        }
                    }
                    double rjj = 0;
                    for (int64_t kk = 0; kk < dim; kk++){
                        rjj += qq[jj][kk] * qq[jj][kk];
                    }
                    rjj = sqrt(rjj);
                    if (rjj > 0){
                        for (int64_t kk = 0; kk < dim; kk++){
                            qq[jj][kk] /= rjj;
                        }
                    }
                    if (jj >= 1){
                        gali *= rjj;
                        GhostTangentData_set_gali(
                            state, ii * (n_ghosts - 1) + jj - 1, gali);
                    }
                }
            }

            // Rescale the ghosts back to distance eps, keeping directions
            for (int64_t gg = 0; gg < n_ghosts; gg++){
                int64_t const slot_g = GhostTangentData_get_slot_of_id(
                    state, n_ref + ii * n_ghosts + gg);
                double const scale = eps / rr[gg];
                for (int64_t kk = 0; kk < dim; kk++){
                    GhostTangent_set_coord(particles, slot_g, kk,
                                           ref[kk] + scale * dphys[gg][kk]);
                }
            }
        }
    }

    END_VECTORIZE;
}


// Metric distance of each tracked particle to stored coordinates
// coords0[particle_id * dim + k]; out[particle_id] = -1 for lost particles.
GPUKERN
void GhostTangent_distance(
    GhostTangentData state,
    ParticlesData particles,
    GPUGLMEM double* coords0,
    GPUGLMEM double* out,
    int64_t n_ids,
    int64_t n_slots)
{
    int64_t const dim = GhostTangentData_get_dim(state);

    VECTORIZE_OVER(slot, n_slots);
        int64_t const pid = ParticlesData_get_particle_id(particles, slot);
        if (pid >= 0 && pid < n_ids){
            if (ParticlesData_get_state(particles, slot) <= 0){
                out[pid] = -1.0;
            }
            else{
                double dd[GHOST_TANGENT_MAX_DIM];
                double nn[GHOST_TANGENT_MAX_DIM];
                for (int64_t kk = 0; kk < dim; kk++){
                    dd[kk] = GhostTangent_get_coord(particles, slot, kk)
                             - coords0[pid * dim + kk];
                }
                out[pid] = GhostTangent_apply_metric(state, dim, dd, nn);
            }
        }
    END_VECTORIZE;
}

#endif /* XTRACK_GHOST_TANGENT_H */
