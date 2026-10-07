# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #
"""Pure-NumPy oracle for the Henon map and the chaos indicators.

Everything here works in normalised coordinates, on momentum
(``Henonmap(norm=True)`` with ``delta = 0``). Phase-space states are arrays of
shape ``(4, N)`` ordered as ``(x, px, y, py)``. The kick is computed from the
complex polynomial ``F(z) = sum_n k_n z^n / n!`` (``z = x + i y``) as
``f_x = Re F``, ``f_y = -Im F``, independently of the monomial expansion used
by the element.
"""

from math import factorial

import numpy as np


class HenonReference:
    """One-turn Henon map ``M = R(omega) o K``: kick, then rotation."""

    def __init__(self, omega_x, omega_y, multipole_coeffs):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.multipole_coeffs = list(multipole_coeffs)
        self.cx, self.sx = np.cos(omega_x), np.sin(omega_x)
        self.cy, self.sy = np.cos(omega_y), np.sin(omega_y)

    # Kick ---------------------------------------------------------------
    def _poly(self, z, deriv=0):
        out = np.zeros_like(z, dtype=complex)
        for n, kn in enumerate(self.multipole_coeffs, start=2):
            if n - deriv < 0:
                continue
            out = out + kn * z**(n - deriv) / factorial(n - deriv)
        return out

    def kick(self, x, y):
        F = self._poly(x + 1j * y)
        return F.real, -F.imag

    def kick_jacobian(self, x, y):
        """Return (dfx/dx, dfx/dy, dfy/dx, dfy/dy)."""
        g = self._poly(x + 1j * y, deriv=1)
        return g.real, -g.imag, -g.imag, -g.real

    # Map ----------------------------------------------------------------
    def step(self, state):
        x, px, y, py = state
        fx, fy = self.kick(x, y)
        pxk = px + fx
        pyk = py + fy
        return np.array([
            self.cx * x + self.sx * pxk,
            -self.sx * x + self.cx * pxk,
            self.cy * y + self.sy * pyk,
            -self.sy * y + self.cy * pyk,
        ])

    def step_inverse(self, state):
        x, px, y, py = state
        xr = self.cx * x - self.sx * px
        pxr = self.sx * x + self.cx * px
        yr = self.cy * y - self.sy * py
        pyr = self.sy * y + self.cy * py
        fx, fy = self.kick(xr, yr)
        return np.array([xr, pxr - fx, yr, pyr - fy])

    def jacobian(self, state):
        """Exact one-turn Jacobian, shape (N, 4, 4)."""
        x, px, y, py = state
        n_part = np.shape(x)[0]
        axx, axy, ayx, ayy = self.kick_jacobian(x, y)
        K = np.zeros((n_part, 4, 4))
        K[:, 0, 0] = 1
        K[:, 1, 1] = 1
        K[:, 2, 2] = 1
        K[:, 3, 3] = 1
        K[:, 1, 0] = axx
        K[:, 1, 2] = axy
        K[:, 3, 0] = ayx
        K[:, 3, 2] = ayy
        R = np.zeros((4, 4))
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = self.cx, self.sx, -self.sx, self.cx
        R[2, 2], R[2, 3], R[3, 2], R[3, 3] = self.cy, self.sy, -self.sy, self.cy
        return R[None, :, :] @ K

    def track(self, state, n_turns, backtrack=False):
        state = np.array(state, dtype=float)
        stepper = self.step_inverse if backtrack else self.step
        for _ in range(n_turns):
            state = stepper(state)
        return state

    def track_modulated(self, state, turns, sin_x, cos_x, sin_y, cos_y,
                        start_turn=0):
        """Track through the given turn numbers with a periodic tune
        modulation: turn t rotates by the angles of index
        (t - start_turn) mod L of the tables."""
        state = np.array(state, dtype=float)
        period = len(sin_x)
        for tt in turns:
            ii = (tt - start_turn) % period
            x, px, y, py = state
            fx, fy = self.kick(x, y)
            pxk = px + fx
            pyk = py + fy
            state = np.array([
                cos_x[ii] * x + sin_x[ii] * pxk,
                -sin_x[ii] * x + cos_x[ii] * pxk,
                cos_y[ii] * y + sin_y[ii] * pyk,
                -sin_y[ii] * y + cos_y[ii] * pyk,
            ])
        return state

    def trajectory(self, state, n_turns):
        """States at turns 0..n_turns, shape (n_turns + 1, 4, N)."""
        state = np.array(state, dtype=float)
        out = [state]
        for _ in range(n_turns):
            state = self.step(state)
            out.append(state)
        return np.array(out)


# Indicators --------------------------------------------------------------

def birkhoff_weights(n):
    """Weights w_k, k = 1..n, of a weighted Birkhoff average over n terms.

    ``w_k ∝ exp(-1 / (t_k (1 - t_k)))`` with ``t_k = k / (n + 1)``, normalised
    to ``sum(w) = 1``.
    """
    t = np.arange(1, n + 1) / (n + 1)
    w = np.exp(-1.0 / (t * (1.0 - t)))
    return w / w.sum()


def renormalisation_turns(num_turns, renorm_every, sample_turns):
    """Turns at which the deviation vectors are renormalised."""
    turns = set(range(renorm_every, num_turns + 1, renorm_every))
    turns |= set(int(tt) for tt in sample_turns)
    turns.add(num_turns)
    return np.array(sorted(turns))


def gram_schmidt_diagonal(vectors):
    """|R_jj| of a modified Gram-Schmidt QR of the columns of `vectors`.

    `vectors` has shape (N, dim, k); returns shape (N, k).
    """
    q = np.array(vectors, dtype=float)
    n_vec = q.shape[2]
    diag = np.zeros((q.shape[0], n_vec))
    for jj in range(n_vec):
        for ii in range(jj):
            proj = np.sum(q[:, :, ii] * q[:, :, jj], axis=1)
            q[:, :, jj] -= proj[:, None] * q[:, :, ii]
        nrm = np.linalg.norm(q[:, :, jj], axis=1)
        diag[:, jj] = nrm
        q[:, :, jj] /= np.where(nrm > 0, nrm, 1.0)[:, None]
    return diag


def sali_from_unit_vectors(u0, u1):
    """SALI = min(|u0 + u1|, |u0 - u1|), vectors of shape (N, dim)."""
    return np.minimum(np.linalg.norm(u0 + u1, axis=1),
                      np.linalg.norm(u0 - u1, axis=1))


def tangent_indicators(ref, z0, num_turns, sample_turns, n_vectors=4,
                       renorm_every=1, initial_vectors=None):
    """Tangent-map indicators computed with the exact Jacobian.

    The deviation vectors start as the first `n_vectors` canonical unit
    vectors (or `initial_vectors`, shape (dim, n_vectors)) and are
    renormalised to unit length, without orthogonalisation, at
    ``renormalisation_turns(num_turns, renorm_every, sample_turns)``.
    At a renormalisation after a chunk of L turns, with stretch factors r_g:

    - ``log_growth[g] += log(r_g)``; FLI = log_growth[0]
    - ``fli_birkhoff[s] += mean(w_{n_s}[chunk]) * log(r_0)`` for chunks within
      the horizon n_s (``w`` from :func:`birkhoff_weights`); this is
      ``sum_k w_k log(r_k)`` exactly when L = 1 and its natural approximation
      otherwise (``log(r_0)`` covers L turns)
    - SALI and GALI_k (k = 2..n_vectors) from the unit vectors.

    Returns a dict of arrays with a leading axis over `sample_turns`:
    ``fli`` (S, N), ``log_growth`` (S, N, G), ``fli_birkhoff`` (S, N),
    ``sali`` (S, N), ``gali`` (S, N, G - 1) with ``gali[..., k - 2] = GALI_k``.
    """
    z = np.array(z0, dtype=float)
    n_part = z.shape[1]
    sample_turns = [int(tt) for tt in sample_turns]
    if initial_vectors is None:
        initial_vectors = np.eye(4)[:, :n_vectors]
    V = np.tile(np.asarray(initial_vectors, dtype=float)[None], (n_part, 1, 1))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    weights = {s: birkhoff_weights(s) for s in sample_turns}
    log_growth = np.zeros((n_part, n_vectors))
    fli_wb = np.zeros((len(sample_turns), n_part))
    sali = np.full(n_part, np.sqrt(2.0))
    gali = np.ones((n_part, n_vectors - 1))
    out = {kk: [] for kk in ('fli', 'log_growth', 'fli_birkhoff', 'sali', 'gali')}

    turn = 0
    for t_next in renormalisation_turns(num_turns, renorm_every, sample_turns):
        chunk = t_next - turn
        for _ in range(chunk):
            V = ref.jacobian(z) @ V
            z = ref.step(z)
        r = np.linalg.norm(V, axis=1)
        log_growth += np.log(r)
        for ss, s_turn in enumerate(sample_turns):
            if t_next <= s_turn:
                w = weights[s_turn][turn:t_next].mean()
                fli_wb[ss] += w * np.log(r[:, 0])
        U = V / r[:, None, :]
        if n_vectors >= 2:
            sali = sali_from_unit_vectors(U[:, :, 0], U[:, :, 1])
            gali = np.cumprod(gram_schmidt_diagonal(U), axis=1)[:, 1:]
        V = U
        turn = t_next
        if turn in sample_turns:
            ss = sample_turns.index(turn)
            out['fli'].append(log_growth[:, 0].copy())
            out['log_growth'].append(log_growth.copy())
            out['fli_birkhoff'].append(fli_wb[ss].copy())
            out['sali'].append(sali.copy())
            out['gali'].append(gali.copy())
    return {kk: np.array(vv) for kk, vv in out.items()}


def reversibility_error(ref, z0, n_turns):
    """REM: distance between z0 and n turns forward then n turns backward."""
    z = ref.track(z0, n_turns)
    zb = ref.track(z, n_turns, backtrack=True)
    return np.linalg.norm(zb - np.asarray(z0), axis=0)


def phases(traj):
    """Normalised phases theta = atan2(-p, q) per plane, from a trajectory
    of shape (T, 4, N); returns (theta_x, theta_y), each (T, N)."""
    return (np.arctan2(-traj[:, 1], traj[:, 0]),
            np.arctan2(-traj[:, 3], traj[:, 2]))


def birkhoff_tunes(traj, start_turn, window):
    """Birkhoff-averaged tunes over two consecutive windows.

    The phase advance of turn j is ``mod(theta_j - theta_{j-1}, 2 pi)``.
    Window 1 averages the advances of turns start_turn+1 .. start_turn+window,
    window 2 those of the next `window` turns, with :func:`birkhoff_weights`.
    Returns ``(qx1, qy1, qx2, qy2, diffusion)`` with
    ``diffusion = log10(sqrt((qx1 - qx2)^2 + (qy1 - qy2)^2))``.
    """
    w = birkhoff_weights(window)
    out = []
    for theta in phases(traj):
        adv = np.mod(np.diff(theta, axis=0), 2 * np.pi)
        a1 = adv[start_turn:start_turn + window]
        a2 = adv[start_turn + window:start_turn + 2 * window]
        out.append((np.sum(w[:, None] * a1, axis=0) / (2 * np.pi),
                    np.sum(w[:, None] * a2, axis=0) / (2 * np.pi)))
    (qx1, qx2), (qy1, qy2) = out
    with np.errstate(divide='ignore'):
        diffusion = np.log10(np.sqrt((qx1 - qx2)**2 + (qy1 - qy2)**2))
    return qx1, qy1, qx2, qy2, diffusion
