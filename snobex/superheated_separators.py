"""
snobex.superheated_separators
=============================
Contours and limiting solutions of superheated transitions, one set per
solution type in ``snobex.superheated``.  Each is derived for the
superheated solution space; none is carried over from the supercooled case.

* ``inverse_jouguet_contour_finder``: the inverse Jouguet deflagrations,
  the fastest walls with the centre at rest.
* ``limiting_inverse_def_contour_finder``: the limiting inverse
  deflagrations, whose rarefaction reaches rest above the degenerate
  critical point (``dcp_energy``) with an infinite slope; they bound the
  inverse deflagrations from above.
"""

import numpy as np
from scipy.optimize import brentq

from .hydrodynamics import integrate_flow
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

    This is exactly the same bound as the one that Mikel and company
    found in 2406.14450

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


def fundamental_derivative(branch, e):
    """
    G(e) = 2 c_s^2 (1 - c_s^2) + (e + p) dc_s^2/de on ``branch``, which has
    the sign of the fundamental derivative of the fluid.  Where G < 0 the
    fluid is not convex: a rarefaction that reaches rest there is
    multivalued, and on the sonic curve mu = c_s the flow turns back.
    """
    c2 = np.asarray(branch.cs2(e), dtype=float)
    dc2 = np.asarray(branch.cs2(e, nu=1), dtype=float)
    return 2.0 * c2 * (1.0 - c2) + (e + np.asarray(branch.p(e), dtype=float)) * dc2


def dcp_energy(branch, n_grid=20001):
    """
    Energy e* of the degenerate critical point (DCP) of ``branch``, where G
    vanishes (arXiv:2607.27874, Eq. (31)), taking the zero closest to the
    turning point.  On the LT branch G < 0 between e* and eTL, since c_s^2
    drops to zero there.  NaN if G never vanishes on the branch.
    """
    top = branch.e_hi if np.isfinite(branch.e_hi) else 10.0 * max(branch.e_turn, 1.0)
    e = np.linspace(branch.e_lo, top, n_grid)[1:-1]
    G = fundamental_derivative(branch, e)
    idx = np.where(np.isfinite(G[:-1]) & np.isfinite(G[1:]) & (G[:-1] * G[1:] < 0))[0]
    if not idx.size:
        return np.nan
    i = idx[0] if branch.up else idx[-1]
    return float(brentq(lambda x: float(fundamental_derivative(branch, x)), e[i], e[i + 1]))


def inverse_wall_residual(xi, v, ep, outside, inside):
    """
    Junction conditions for a wall at ``xi`` with the centre at rest, given
    the outside state (plasma-frame velocity ``v``, energy ``ep``) in front
    of it.  With a = |v+| = |v - xi| / (1 - v xi) and v- = -xi, they are two
    straight lines in the (E_C, p-) plane,
        momentum:     p- = p+ - xi a (e+ - E_C),
        energy flux:  p- = (xi / a)(E_C + p+) - e+,
    and the wall needs their intersection to lie on the inside branch.

    Returns
    -------
    (residual, eC) : p_in(E_C) - p- at the intersection and its E_C; the
        residual is NaN if E_C is off the inside branch.
    """
    a = abs((v - xi) / (1.0 - v * xi))
    pp = float(outside.p(ep))
    b1, a1 = xi * a, pp - xi * a * ep
    b2, a2 = xi / a, xi * pp / a - ep
    if abs(b1 - b2) < 1e-15:
        return np.nan, np.nan
    eC = (a2 - a1) / (b1 - b2)
    if not inside.e_lo <= eC <= inside.e_hi:
        return np.nan, eC
    return float(inside.p(eC)) - (a1 + b1 * eC), eC


def limiting_inverse_def_contour_finder(xi0, outside, inside, e_star=None, v_start=1e-8,
                                        t_max=1e7, xi_min=1e-6, n_side=200):
    """
    Limiting inverse deflagration whose rarefaction reaches rest at
    xi0 = c_s(E_N) with dv/dxi -> infinity (the limiting solutions of the
    second kind of arXiv:2607.27874, Sec. 5.2, for a rarefaction in front of
    the wall).

    Above the DCP, E_N > e*, every rarefaction reaching rest turns back once
    (the fluid is not convex there), with a fold of any size.  The limiting
    solution has its fold at the point of rest, of size ``v_start``: it is
    started on the sonic curve mu(xi0, -v_start) = c_s(E_N) and integrated
    towards the centre until the junction conditions with the centre at rest
    hold, which places the wall.  Sweeping xi0 in (0, c_s(e*)) gives the
    separator that bounds the inverse deflagrations from above; it closes at
    xi_w -> 0, E_N -> eTL.

    The curve moves (logarithmically) with ``v_start``.  The default equals
    the ``v_rest`` of ``find_inverse_def``, so that it bounds the solutions
    the finder accepts.  There are no limiting solutions of the first kind
    (the DCP inside the rarefaction): the flow through the DCP touches the
    sonic curve from the supersonic side, since there d^2(mu - c_s)/dt^2 is
    proportional to -v^2 G'(e*) > 0.

    Parameters
    ----------
    xi0     : float   Point of rest, xi0 = c_s(E_N); below c_s(e*).
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).
    e_star  : float   DCP of the outside branch (computed if None).
    v_start : float   Size of the fold at the point of rest.
    t_max   : float   Maximum integration length in the flow parameter t; near
                      eTL the flow evolves on times ~ 1 / c_s^2.
    xi_min  : float   The flow towards the centre is followed down to xi_min.
    n_side  : int     Points of the regions at rest in the returned flow.

    Returns
    -------
    (xiw, eN, eC, flow) : floats and [v(xi), xi, e(xi)] from the centre to
        xi = 1 (as returned by ``find_inverse_def``), or NaNs.
    """
    fail = (np.nan, np.nan, np.nan, np.nan)
    xi0 = float(xi0.item() if np.ndim(xi0) > 0 else xi0)
    if e_star is None:
        e_star = dcp_energy(outside)
    if not np.isfinite(e_star):
        return fail

    # point of rest: E_N between the DCP and the turning point, on the sonic curve
    v0 = -v_start
    mu = (xi0 - v0) / (1.0 - xi0 * v0)
    edge = outside.e_turn + (1e-9 if outside.up else -1e-9) * abs(outside.e_turn - e_star)
    try:
        eN = brentq(lambda e: np.sqrt(max(float(outside.cs2(e)), 0.0)) - mu, *sorted((e_star, edge)))
    except ValueError:
        return fail

    # rarefaction towards the centre, until the junction conditions hold
    def centre(t, y, *args):
        return y[1] - xi_min
    def beyond(t, y, *args):
        return y[1] - 1.0
    centre.terminal = beyond.terminal = True
    sol = integrate_flow(outside, [v0, xi0, eN], (0.0, t_max), events=[centre, beyond],
                         rtol=1e-10, atol=[1e-4 * v_start, 1e-12, 1e-12], dense_output=True)
    t_end = sol.t[-1]
    ts = np.unique(np.concatenate((np.linspace(0.0, t_end, 1000),
                                   np.geomspace(min(1e-6 * t_end, 1e-3), t_end, 1000))))
    v, xi, e = sol.sol(ts)
    R = np.array([inverse_wall_residual(x, u, w, outside, inside)[0] for u, x, w in zip(v, xi, e)])
    cross = np.where(np.isfinite(R[:-1]) & np.isfinite(R[1:]) & (R[:-1] * R[1:] < 0))[0]
    if not cross.size:
        return fail
    k = cross[0]
    resid = lambda t: inverse_wall_residual(*(lambda y: (y[1], y[0], y[2]))(sol.sol(t)), outside, inside)[0]
    t_w = brentq(resid, ts[k], ts[k + 1], xtol=1e-12)
    vw, xw, ew = sol.sol(t_w)
    _, eC = inverse_wall_residual(xw, vw, ew, outside, inside)

    # an inverse deflagration: single-valued subsonic rarefaction, inside subsonic
    v, xi, e = (np.append(a[:k + 1], b) for a, b in zip((v, xi, e), (vw, xw, ew)))
    vp = (vw - xw) / (1.0 - vw * xw)
    if (np.any(np.diff(xi) > 1e-12 * xi0) or not vp**2 < float(outside.cs2(ew))
            or not xw**2 < float(inside.cs2(eC))):
        return fail
    flow = [np.concatenate((np.zeros(n_side), v[::-1], np.zeros(n_side))),
            np.concatenate((np.linspace(0.0, xw, n_side), xi[::-1], np.linspace(xi0, 1.0, n_side))),
            np.concatenate((np.full(n_side, eC), e[::-1], np.full(n_side, eN)))]
    return float(xw), float(eN), float(eC), flow
