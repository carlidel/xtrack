// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_HENONMAP_H
#define XTRACK_HENONMAP_H

#ifdef XO_CONTEXT_CPU
#include <math.h>
#endif  // XO_CONTEXT_CPU

#include "xtrack/headers/track.h"

// Maximum number of monomials per plane, enforced in Henonmap.__init__.
#define HENONMAP_MAX_COEFFS 128


GPUFUN
double Henonmap_eval_poly(
    const double* coeffs, const int64_t* x_exps, const int64_t* y_exps,
    int64_t n_coeffs, double scale, double xx, double yy)
{
    double res = 0;
    for (int64_t i = 0; i < n_coeffs; i++){
        double prod = coeffs[i] * scale;
        for (int64_t j = 0; j < x_exps[i]; j++){
            prod *= xx;
        }
        for (int64_t j = 0; j < y_exps[i]; j++){
            prod *= yy;
        }
        res += prod;
    }
    return res;
}


GPUFUN
void Henonmap_track_local_particle(HenonmapData el, LocalParticle* part0){

    double const sin_omega_x = HenonmapData_get_sin_omega_x(el);
    double const cos_omega_x = HenonmapData_get_cos_omega_x(el);
    double const sin_omega_y = HenonmapData_get_sin_omega_y(el);
    double const cos_omega_y = HenonmapData_get_cos_omega_y(el);

    int64_t const n_turns = HenonmapData_get_n_turns(el);

    int64_t n_fx_coeffs = HenonmapData_get_n_fx_coeffs(el);
    int64_t n_fy_coeffs = HenonmapData_get_n_fy_coeffs(el);
    if (n_fx_coeffs > HENONMAP_MAX_COEFFS) n_fx_coeffs = HENONMAP_MAX_COEFFS;
    if (n_fy_coeffs > HENONMAP_MAX_COEFFS) n_fy_coeffs = HENONMAP_MAX_COEFFS;

    double const alpha_x = HenonmapData_get_twiss_params(el, 0);
    double const beta_x = HenonmapData_get_twiss_params(el, 1);
    double const alpha_y = HenonmapData_get_twiss_params(el, 2);
    double const beta_y = HenonmapData_get_twiss_params(el, 3);
    double const sqrt_beta_x = sqrt(beta_x);
    double const sqrt_beta_y = sqrt(beta_y);
    double const domegax = HenonmapData_get_domegax(el);
    double const domegay = HenonmapData_get_domegay(el);
    double const dx = HenonmapData_get_dx(el);
    double const ddx = HenonmapData_get_ddx(el);

    int64_t const norm = HenonmapData_get_norm(el);
    int64_t const mod_period = HenonmapData_get_modulation_period(el);
    int64_t const mod_start = HenonmapData_get_modulation_start_turn(el);

    int const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    double fx_coeffs[HENONMAP_MAX_COEFFS];
    int64_t fx_x_exps[HENONMAP_MAX_COEFFS];
    int64_t fx_y_exps[HENONMAP_MAX_COEFFS];
    for (int64_t i = 0; i < n_fx_coeffs; i++){
        fx_coeffs[i] = HenonmapData_get_fx_coeffs(el, i);
        fx_x_exps[i] = HenonmapData_get_fx_x_exps(el, i);
        fx_y_exps[i] = HenonmapData_get_fx_y_exps(el, i);
    }

    double fy_coeffs[HENONMAP_MAX_COEFFS];
    int64_t fy_x_exps[HENONMAP_MAX_COEFFS];
    int64_t fy_y_exps[HENONMAP_MAX_COEFFS];
    for (int64_t i = 0; i < n_fy_coeffs; i++){
        fy_coeffs[i] = HenonmapData_get_fy_coeffs(el, i);
        fy_x_exps[i] = HenonmapData_get_fy_x_exps(el, i);
        fy_y_exps[i] = HenonmapData_get_fy_y_exps(el, i);
    }

    START_PER_PARTICLE_BLOCK(part0, part);

        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double const delta = LocalParticle_get_delta(part);

        double x_hat, px_hat, y_hat, py_hat;
        if (norm){
            x_hat = x;
            px_hat = px;
            y_hat = y;
            py_hat = py;
        }
        else{
            x_hat = x / sqrt_beta_x;
            px_hat = alpha_x * x / sqrt_beta_x + px * sqrt_beta_x;
            y_hat = y / sqrt_beta_y;
            py_hat = alpha_y * y / sqrt_beta_y + py * sqrt_beta_y;
        }
        double const x_hat_f = dx * delta / sqrt_beta_x;
        double const px_hat_f = alpha_x * dx * delta / sqrt_beta_x
                                + ddx * delta * sqrt_beta_x;

        // Linear tunes of this turn: static, or from the periodic
        // modulation tables (index (at_turn - start) mod period, also in
        // backtracking, where at_turn is decremented before the turn)
        double base_cos_omega_x = cos_omega_x;
        double base_sin_omega_x = sin_omega_x;
        double base_cos_omega_y = cos_omega_y;
        double base_sin_omega_y = sin_omega_y;
        if (mod_period > 0){
            int64_t idx = (LocalParticle_get_at_turn(part) - mod_start) % mod_period;
            if (idx < 0) idx += mod_period;
            base_sin_omega_x = HenonmapData_get_modulation_sin_omega_x(el, idx);
            base_cos_omega_x = HenonmapData_get_modulation_cos_omega_x(el, idx);
            base_sin_omega_y = HenonmapData_get_modulation_sin_omega_y(el, idx);
            base_cos_omega_y = HenonmapData_get_modulation_cos_omega_y(el, idx);
        }

        double curr_cos_omega_x = base_cos_omega_x;
        double curr_sin_omega_x = base_sin_omega_x;
        double curr_cos_omega_y = base_cos_omega_y;
        double curr_sin_omega_y = base_sin_omega_y;
        if (domegax != 0){
            double const cos_domega_x = cos(domegax * delta);
            double const sin_domega_x = sin(domegax * delta);
            curr_cos_omega_x = base_cos_omega_x * cos_domega_x - base_sin_omega_x * sin_domega_x;
            curr_sin_omega_x = base_sin_omega_x * cos_domega_x + base_cos_omega_x * sin_domega_x;
        }
        if (domegay != 0){
            double const cos_domega_y = cos(domegay * delta);
            double const sin_domega_y = sin(domegay * delta);
            curr_cos_omega_y = base_cos_omega_y * cos_domega_y - base_sin_omega_y * sin_domega_y;
            curr_sin_omega_y = base_sin_omega_y * cos_domega_y + base_cos_omega_y * sin_domega_y;
        }

        double const multipole_scale = 1.0 / (1.0 + delta);

        for (int64_t n = 0; n < n_turns; n++){

            if (backtrack){
                // Exact inverse: undo the rotation, then undo the kick
                double const xr = x_hat - x_hat_f;
                double const pr = px_hat - px_hat_f;
                x_hat = curr_cos_omega_x * xr - curr_sin_omega_x * pr + x_hat_f;
                px_hat = curr_sin_omega_x * xr + curr_cos_omega_x * pr + px_hat_f;
                double const y_hat_new = curr_cos_omega_y * y_hat - curr_sin_omega_y * py_hat;
                double const py_hat_new = curr_sin_omega_y * y_hat + curr_cos_omega_y * py_hat;
                y_hat = y_hat_new;
                py_hat = py_hat_new;
            }

            double const xx = sqrt_beta_x * x_hat;
            double const yy = sqrt_beta_y * y_hat;
            double const fx = sqrt_beta_x * Henonmap_eval_poly(
                fx_coeffs, fx_x_exps, fx_y_exps, n_fx_coeffs, multipole_scale, xx, yy);
            double const fy = sqrt_beta_y * Henonmap_eval_poly(
                fy_coeffs, fy_x_exps, fy_y_exps, n_fy_coeffs, multipole_scale, xx, yy);

            if (backtrack){
                px_hat -= fx;
                py_hat -= fy;
            }
            else{
                // Kick, then rotation (clockwise in the normalised phase space)
                double const xr = x_hat - x_hat_f;
                double const pr = px_hat - px_hat_f + fx;
                double const x_hat_new = curr_cos_omega_x * xr + curr_sin_omega_x * pr + x_hat_f;
                double const px_hat_new = -curr_sin_omega_x * xr + curr_cos_omega_x * pr + px_hat_f;
                double const y_hat_new = curr_cos_omega_y * y_hat + curr_sin_omega_y * (py_hat + fy);
                double const py_hat_new = -curr_sin_omega_y * y_hat + curr_cos_omega_y * (py_hat + fy);
                x_hat = x_hat_new;
                px_hat = px_hat_new;
                y_hat = y_hat_new;
                py_hat = py_hat_new;
            }
        }

        if (norm){
            x = x_hat;
            px = px_hat;
            y = y_hat;
            py = py_hat;
        }
        else{
            x = sqrt_beta_x * x_hat;
            px = -alpha_x * x_hat / sqrt_beta_x + px_hat / sqrt_beta_x;
            y = sqrt_beta_y * y_hat;
            py = -alpha_y * y_hat / sqrt_beta_y + py_hat / sqrt_beta_y;
        }

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);

    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_HENONMAP_H */
