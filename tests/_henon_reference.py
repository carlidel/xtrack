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
