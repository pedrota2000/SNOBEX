"""
snobex.superheated_separators
=============================
Contours and limiting solutions of superheated transitions, one set per
solution type in ``snobex.superheated``.  Each is derived for the
superheated solution space; none is carried over from the supercooled case.

* ``inverse_jouguet_contour_finder``: the inverse Jouguet deflagrations,
  the fastest walls with the centre at rest.
"""

import numpy as np
from scipy.optimize import brentq

from .superheated import find_inverse_def


def inverse_jouguet_contour_finder(eC, outside, inside, n_grid=4000):
    """
    Inverse Jouguet deflagration for centre energy ``eC``.

    The flow in front of the wall moves at the sound speed, v+ = -c_s(e+).
    With the centre at rest (v- = -xi_w, e- = E_C) the junction conditions
        v+ / v- = (E_C + p+) / (e+ + p-),   v+ v- = (p+ - p-) / (e+ - E_C)
    then give
        c_s^2(e+) (e+ + p-)(e+ - E_C) = (p+ - p-)(E_C + p+),
        xi_w = c_s(e+) (e+ + p-) / (E_C + p+).
    This is the largest xi_w for which the junction conditions have a real
    solution on the outside branch: there the subsonic and the supersonic
    wall states merge, and for faster walls they are complex.  So the curve
    bounds the inverse deflagrations, as the Jouguet detonations bound the
    detonations from below.

    Parameters
    ----------
    eC      : float   Energy at the centre of the bubble (inside branch).
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).
    n_grid  : int     Points of the scan for the sonic wall state.

    Returns
    -------
    (xiw, eN) : (float, float) or (NaN, NaN)
        eN is NaN where the rarefaction in front of the Jouguet wall does
        not reach rest as a single-valued profile.
    """
    eC = float(eC.item() if np.ndim(eC) > 0 else eC)
    pm = float(inside.p(eC))
    e_hi = outside.e_hi if np.isfinite(outside.e_hi) else 10.0 * max(outside.e_turn, 1.0)
    grid = np.linspace(outside.e_lo, e_hi, n_grid)[1:-1]

    def sonic(e):
        P = np.asarray(outside.p(e), dtype=float)
        return np.asarray(outside.cs2(e), dtype=float) * (e + pm) * (e - eC) - (P - pm) * (eC + P)

    S = sonic(grid)
    best = None
    for i in np.where(np.isfinite(S[:-1]) & np.isfinite(S[1:]) & (S[:-1] * S[1:] < 0))[0]:
        ep = brentq(lambda e: float(sonic(e)), grid[i], grid[i + 1], xtol=1e-14)
        cs = np.sqrt(max(float(outside.cs2(ep)), 0.0))
        xiw = cs * (ep + pm) / (eC + float(outside.p(ep)))
        # the fluid in front must move inwards, |v+| > xi_w; the maximum of
        # xi_w over the branch is the Jouguet state
        if 0.0 < xiw < cs < 1.0 and (best is None or xiw > best[0]):
            best = (xiw, ep, -cs)
    if best is None:
        return np.nan, np.nan
    xiw, ep, vp = best
    eN, _ = find_inverse_def(eC, xiw, outside, inside, wall=(vp, ep))
    return float(xiw), float(eN)
