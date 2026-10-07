# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #

from math import factorial

import numpy as np
import xobjects as xo

from ..base_element import BeamElement

# Size of the local coefficient arrays in henonmap.h
HENONMAP_MAX_COEFFS = 128


def _henon_polynomial_coefficients(multipole_coeffs):
    """Expand normal multipole strengths into the monomials of the kick.

    The kick of order ``n`` (``n >= 2``) with strength ``k`` is
    ``Re[k (x + i y)^n / n!]`` on ``px`` and ``-Im[k (x + i y)^n / n!]`` on
    ``py``. Returns ``(fx_coeffs, fx_x_exps, fx_y_exps, fy_coeffs, fy_x_exps,
    fy_y_exps)``.
    """
    fx_coeffs, fx_x_exps, fx_y_exps = [], [], []
    fy_coeffs, fy_x_exps, fy_y_exps = [], [], []
    for n in range(2, len(multipole_coeffs) + 2):
        kn = multipole_coeffs[n - 2]
        for k in range(0, n + 1):
            cc = kn / factorial(k) / factorial(n - k)
            if (k % 4) == 0:
                fx_coeffs.append(cc)
                fx_x_exps.append(n - k)
                fx_y_exps.append(k)
            elif (k % 4) == 2:
                fx_coeffs.append(-cc)
                fx_x_exps.append(n - k)
                fx_y_exps.append(k)
            elif (k % 4) == 1:
                fy_coeffs.append(-cc)
                fy_x_exps.append(n - k)
                fy_y_exps.append(k)
            else:
                fy_coeffs.append(cc)
                fy_x_exps.append(n - k)
                fy_y_exps.append(k)
    return fx_coeffs, fx_x_exps, fx_y_exps, fy_coeffs, fy_x_exps, fy_y_exps


class Henonmap(BeamElement):
    '''Beam element representing a Henon-like map with an arbitrary polynomial
    kick, i.e. a one-turn map made of a thin normal multipole kick followed by
    a linear rotation in the normalised phase space.

    In normalised coordinates (``norm=True``, on momentum) one turn reads::

        px' = px + f_x(x, y),   py' = py + f_y(x, y)
        (x, px) <- R(omega_x) (x, px'),   (y, py) <- R(omega_y) (y, py')

    with ``R(w) = [[cos w, sin w], [-sin w, cos w]]``,
    ``f_x = Re sum_n k_n (x + i y)^n / n!`` and
    ``f_y = -Im sum_n k_n (x + i y)^n / n!`` (``n >= 2``).
    For example ``multipole_coeffs=[2, 6 * mu]`` gives
    ``f_x = x^2 - y^2 + mu (x^3 - 3 x y^2)`` and
    ``f_y = -2 x y + mu (y^3 - 3 x^2 y)``.

    The element has an exact inverse and supports backtracking.

    Parameters
    ----------
    omega_x : float
        Linear angular frequency (``2 pi Q_x``) in the horizontal plane.
        Default is ``0``.
    omega_y : float
        Linear angular frequency (``2 pi Q_y``) in the vertical plane.
        Default is ``0``.
    n_turns : int
        Number of turns applied each time the element is tracked. Default is
        ``1``. In general, tracking for multiple turns should be done by
        wrapping the element in a line and providing ``num_turns`` to
        ``Line.track``.
    twiss_params : array of floats
        ``[alpha_x, beta_x, alpha_y, beta_y]`` used for coordinate
        normalisation and denormalisation. Default is ``[0, 1, 0, 1]``.
    dqx : float
        Horizontal chromaticity of the ring. Default is ``0``.
    dqy : float
        Vertical chromaticity of the ring. Default is ``0``.
    dx : float
        Horizontal dispersion at the location of the multipole. Default is
        ``0``.
    ddx : float
        Derivative of the horizontal dispersion at the location of the
        multipole. Default is ``0``.
    multipole_coeffs : array of floats
        Integrated normal multipole strengths in increasing order, starting
        from the sextupole (``k2l``, ``k3l``, ...). Skew components are not
        supported. Default is ``[0]`` (no kick).
    norm : bool
        ``True`` if the particle coordinates are already normalised, ``False``
        otherwise. Default is ``False``.
    modulation_sin_omega_x, modulation_cos_omega_x, modulation_sin_omega_y, \
    modulation_cos_omega_y : arrays of floats, optional
        Turn-by-turn tune modulation: pre-computed ``sin`` and ``cos`` of the
        linear angular frequencies, one value per turn, all of the same
        length ``L`` (the modulation period). When given, they replace
        ``omega_x`` and ``omega_y``. Turn ``t`` (the particle's ``at_turn``)
        uses index ``(t - modulation_start_turn) mod L``: **the modulation
        is periodic with period L turns** and wraps around in both
        directions, so tracking may run for any number of turns. To model a
        quasi-periodic modulation, choose ``L`` as a common period of its
        components or long enough for the run. ``sin^2 + cos^2 = 1`` is
        checked (the map stays symplectic). Chromaticity is applied on top
        of the modulated tunes. Backtracking turn ``t`` uses the same index
        as the forward turn ``t``, so the inverse is exact. Requires
        ``n_turns = 1``. See :meth:`modulation_arrays` to build the arrays
        from per-turn frequencies.
    modulation_start_turn : int
        Turn mapped to index 0 of the modulation arrays. Default ``0``.

    Notes
    -----
    The fields ``sin_omega_x``, ``cos_omega_x``, ``sin_omega_y`` and
    ``cos_omega_y`` store the trigonometric functions of the tunes
    (``modulation_period = 0`` when there is no modulation), and
    ``domegax = 2 pi dqx``, ``domegay = 2 pi dqy``. The polynomial kick is
    stored as monomials ``x^n y^m`` in ``fx_coeffs``, ``fx_x_exps``,
    ``fx_y_exps`` (horizontal) and ``fy_*`` (vertical), at most
    ``HENONMAP_MAX_COEFFS = 128`` per plane.

    Originally developed by Dora Veres and Carlo Emilio Montanari.
    '''

    _xofields = {
        'sin_omega_x': xo.Float64,
        'cos_omega_x': xo.Float64,
        'sin_omega_y': xo.Float64,
        'cos_omega_y': xo.Float64,
        'n_turns': xo.Int64,
        'twiss_params': xo.Float64[:],
        'domegax': xo.Float64,
        'domegay': xo.Float64,
        'dx': xo.Float64,
        'ddx': xo.Float64,
        'fx_coeffs': xo.Float64[:],
        'fx_x_exps': xo.Int64[:],
        'fx_y_exps': xo.Int64[:],
        'fy_coeffs': xo.Float64[:],
        'fy_x_exps': xo.Int64[:],
        'fy_y_exps': xo.Int64[:],
        'n_fx_coeffs': xo.Int64,
        'n_fy_coeffs': xo.Int64,
        'norm': xo.Int64,
        'modulation_period': xo.Int64,
        'modulation_start_turn': xo.Int64,
        'modulation_sin_omega_x': xo.Float64[:],
        'modulation_cos_omega_x': xo.Float64[:],
        'modulation_sin_omega_y': xo.Float64[:],
        'modulation_cos_omega_y': xo.Float64[:],
    }

    _modulation_fields = ('modulation_sin_omega_x', 'modulation_cos_omega_x',
                          'modulation_sin_omega_y', 'modulation_cos_omega_y')

    isthick = False
    behaves_like_drift = False
    has_backtrack = True
    allow_rot_and_shift = False

    # Not part of the prebuilt kernels
    allow_kernel_compilation = True

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/henonmap.h"',
    ]

    def __init__(self, omega_x=0.,
                       omega_y=0.,
                       n_turns=1,
                       twiss_params=None,
                       dqx=0,
                       dqy=0,
                       dx=0,
                       ddx=0,
                       multipole_coeffs=None,
                       norm=False,
                       **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            super().__init__(**kwargs)
            return

        if twiss_params is None:
            twiss_params = [0.0, 1.0, 0.0, 1.0]
        twiss_params = np.asarray(twiss_params, dtype=np.float64)
        if twiss_params.shape != (4,):
            raise ValueError(
                '`twiss_params` must be [alpha_x, beta_x, alpha_y, beta_y]')
        if twiss_params[1] <= 0 or twiss_params[3] <= 0:
            raise ValueError('Beta functions in `twiss_params` must be positive')

        if multipole_coeffs is None:
            multipole_coeffs = [0.0]

        if int(n_turns) < 0:
            raise ValueError('`n_turns` must be non-negative')

        kwargs.setdefault('sin_omega_x', np.sin(omega_x))
        kwargs.setdefault('cos_omega_x', np.cos(omega_x))
        kwargs.setdefault('sin_omega_y', np.sin(omega_y))
        kwargs.setdefault('cos_omega_y', np.cos(omega_y))
        kwargs.setdefault('n_turns', n_turns)
        kwargs.setdefault('twiss_params', twiss_params)
        kwargs.setdefault('domegax', 2 * np.pi * dqx)
        kwargs.setdefault('domegay', 2 * np.pi * dqy)
        kwargs.setdefault('dx', dx)
        kwargs.setdefault('ddx', ddx)

        (fx_coeffs, fx_x_exps, fx_y_exps,
         fy_coeffs, fy_x_exps, fy_y_exps) = _henon_polynomial_coefficients(
            multipole_coeffs)
        kwargs.setdefault('fx_coeffs', fx_coeffs)
        kwargs.setdefault('fx_x_exps', fx_x_exps)
        kwargs.setdefault('fx_y_exps', fx_y_exps)
        kwargs.setdefault('fy_coeffs', fy_coeffs)
        kwargs.setdefault('fy_x_exps', fy_x_exps)
        kwargs.setdefault('fy_y_exps', fy_y_exps)
        kwargs.setdefault('n_fx_coeffs', len(kwargs['fx_coeffs']))
        kwargs.setdefault('n_fy_coeffs', len(kwargs['fy_coeffs']))
        kwargs.setdefault('norm', 1 if norm else 0)

        self._prepare_modulation(kwargs)

        for plane in ('fx', 'fy'):
            nn = kwargs[f'n_{plane}_coeffs']
            if nn > HENONMAP_MAX_COEFFS:
                raise ValueError(
                    f'Too many monomials in the {plane} kick ({nn} > '
                    f'{HENONMAP_MAX_COEFFS}); reduce the multipole order.')
            for name in ('coeffs', 'x_exps', 'y_exps'):
                if len(kwargs[f'{plane}_{name}']) != nn:
                    raise ValueError(
                        f'`{plane}_{name}` must have length n_{plane}_coeffs')

        super().__init__(**kwargs)

    def _prepare_modulation(self, kwargs):
        arrays = [kwargs.get(nn) for nn in self._modulation_fields]
        given = [aa is not None and len(aa) > 0 for aa in arrays]
        kwargs.setdefault('modulation_start_turn', 0)
        if not any(given):
            for nn in self._modulation_fields:
                kwargs[nn] = np.zeros(0)
            kwargs['modulation_period'] = 0
            return
        if not all(given):
            raise ValueError('The four modulation arrays (sin and cos of '
                             'omega_x and omega_y) must be given together')
        arrays = [np.ascontiguousarray(aa, dtype=np.float64) for aa in arrays]
        period = len(arrays[0])
        if any(aa.ndim != 1 or len(aa) != period for aa in arrays):
            raise ValueError('The modulation arrays must be 1D and have the '
                             'same length')
        if kwargs.get('modulation_period', period) != period:
            raise ValueError('`modulation_period` must equal the length of '
                             'the modulation arrays')
        for ss, cc, plane in ((arrays[0], arrays[1], 'x'),
                              (arrays[2], arrays[3], 'y')):
            if np.max(np.abs(ss**2 + cc**2 - 1)) > 1e-12:
                raise ValueError(f'sin^2 + cos^2 != 1 in the {plane} '
                                 'modulation arrays')
        if kwargs['n_turns'] != 1:
            raise ValueError('Tune modulation requires `n_turns = 1`')
        for nn, aa in zip(self._modulation_fields, arrays):
            kwargs[nn] = aa
        kwargs['modulation_period'] = period

    @staticmethod
    def modulation_arrays(omega_x, omega_y):
        """Modulation arrays from per-turn angular frequencies.

        Parameters
        ----------
        omega_x, omega_y : arrays of floats
            ``2 pi Q`` for turns ``0 .. L-1`` of one modulation period, e.g.
            ``omega_x0 * (1 + eps * sum_k eps_k cos(Omega_k n))``.

        Returns
        -------
        dict
            Keyword arguments for :class:`Henonmap`
            (``modulation_sin_omega_x``, ``modulation_cos_omega_x``,
            ``modulation_sin_omega_y``, ``modulation_cos_omega_y``).
        """
        omega_x = np.asarray(omega_x, dtype=np.float64)
        omega_y = np.asarray(omega_y, dtype=np.float64)
        return {'modulation_sin_omega_x': np.sin(omega_x),
                'modulation_cos_omega_x': np.cos(omega_x),
                'modulation_sin_omega_y': np.sin(omega_y),
                'modulation_cos_omega_y': np.cos(omega_y)}

    @property
    def omega_x(self):
        return np.arctan2(self.sin_omega_x, self.cos_omega_x)

    @property
    def omega_y(self):
        return np.arctan2(self.sin_omega_y, self.cos_omega_y)
