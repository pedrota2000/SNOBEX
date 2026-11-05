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
* ``spinodal_inverse_det_contour_finder``: the slowest inverse detonations,
  whose state behind the wall is at the turning point of the inside phase;
  slower walls have no junction state on the inside branch.
* ``sonic_inverse_det_contour_finder``: the inverse detonations whose wall
  moves at the sound speed of the outside phase at rest, the border with
  the inverse hybrids.
* ``inverse_hyb_wall_obstruction`` and ``inverse_hyb_obstruction_energies``:
  the wall obstruction of the inverse hybrids, from the equation of state
  alone (a sonic front with the inside state behind the wall at its turning
  point); with no stretch where it holds, the EoS has no inverse hybrids.
  ``spinodal_inverse_hyb_contour_finder``: the limiting inverse hybrids at
  its roots, swept in the wall velocity.
* ``limiting_inverse_hyb_contour_finder``: the limiting inverse hybrids of
  the second kind, whose rarefaction in front reaches rest above the DCP
  with an infinite slope (the flow obstruction that bounds the hybrids from
  above in E_N).
"""

import numpy as np
from scipy.optimize import brentq

from .hydrodynamics import integrate_flow, rest_behind_residual
from .superheated import (find_inverse_def, find_inverse_det, find_inverse_hyb,
                          _weak_inside_states, _shock_to_rest_behind)


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
    of it: ``rest_behind_residual`` with the outside phase ahead.

    Returns
    -------
    (residual, eC) : p_in(E_C) - p- at the intersection of the two junction
        conditions in the (E_C, p-) plane and its E_C; the residual is NaN if
        E_C is off the inside branch.
    """
    return rest_behind_residual(xi, v, ep, outside, inside)


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


def spinodal_inverse_det_contour_finder(eN, outside, inside):
    """
    Slowest inverse detonation for nucleation energy ``eN``: the state behind
    the wall at the turning point of the inside branch, e- = eTH, where
    c_s,in = 0.

    With the outside at rest (v+ = -xi_w, e+ = E_N) the junction conditions
    give each inside state e- the wall speed
        xi_w^2 = (p+ - p-)(e- + p+) / ((E_N + p-)(E_N - e-)),
        v- = -xi_w (E_N + p-) / (e- + p+).
    For the analytic model xi_w(e-) grows from the turning point, with no
    interior minimum (no Jouguet point, |v-| = c_s, on the branch: every
    wall state is weak).  Slower walls would need a state past the spinodal,
    so the junction conditions have no solution on the inside branch and the
    curve bounds the inverse detonations from below, as the Jouguet
    detonations bound the supercooled ones.  For another EoS check that
    xi_w(e-) has no interior minimum (``scripts/inverse_detonation_profile.py``
    draws it).

    Parameters
    ----------
    eN      : float   Nucleation energy (outside branch).
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).

    Returns
    -------
    (xiw, eC) : (float, float) or (NaN, NaN)
        NaN where that wall is not supersonic with respect to the outside
        phase (xi_w <= c_s,out(E_N)), the fluid behind it does not move
        inwards, or ``find_inverse_det`` finds no shock bringing it to rest.
    """
    eN = float(eN.item() if np.ndim(eN) > 0 else eN)
    em = float(inside.e_turn)
    pp, pm = float(outside.p(eN)), float(inside.p(em))
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (pp - pm) * (em + pp) / ((eN + pm) * (eN - em))
    if not (np.isfinite(h) and 0.0 < h < 1.0):
        return np.nan, np.nan
    xiw = float(np.sqrt(h))
    vm = -xiw * (eN + pm) / (em + pp)
    if not -1.0 < vm < -xiw:
        return np.nan, np.nan
    eC, _ = find_inverse_det(eN, xiw, outside, inside, wall=(vm, em))
    if not np.isfinite(eC):
        return np.nan, np.nan
    return xiw, float(eC)


def sonic_inverse_det_contour_finder(eN, outside, inside):
    """
    Inverse detonation whose wall moves at the sound speed of the outside
    phase at rest, xi_w = c_s,out(E_N): the border between the inverse
    detonations (faster walls, the fluid in front stays at rest) and the
    inverse hybrids (slower walls: the push of the wall runs ahead of it,
    and a rarefaction in front of the wall ends at |v+| = c_s(e+); as xi_w
    rises to c_s,out(E_N) it shrinks to nothing).  The counterpart of the
    supercooled deflagration separator xi_w = c_s,in(E_C)
    (``compute_def_separator``), where the wall moves at the sound speed of
    the phase at rest next to it.

    The profile is the limit of ``find_inverse_det`` from faster walls,
    evaluated at the next float above c_s,out(E_N).

    Parameters
    ----------
    eN      : float   Nucleation energy (outside branch).
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).

    Returns
    -------
    (xiw, eC) : (float, float) or (NaN, NaN)
        NaN where no inverse detonation has that wall speed, e.g. where the
        slowest one (``spinodal_inverse_det_contour_finder``) is faster.
    """
    eN = float(eN.item() if np.ndim(eN) > 0 else eN)
    cs2 = float(outside.cs2(eN))
    if not (np.isfinite(cs2) and 0.0 < cs2 < 1.0):
        return np.nan, np.nan
    xiw = float(np.nextafter(np.sqrt(cs2), 1.0))
    eC, _ = find_inverse_det(eN, xiw, outside, inside)
    if not np.isfinite(eC):
        return np.nan, np.nan
    return xiw, float(eC)


def inverse_hyb_wall_obstruction(e, outside, inside):
    """
    Wall obstruction of the inverse hybrids, from the equation of state alone:
        G(e+) = c_s,out^2(e+) (e+ - e*)(e+ + p1) - (p_out(e+) - p1)(e* + p_out(e+)),
    with e* the inside turning point and p1 = p_in(e*).  It is the junction
    condition v+^2 = (p+ - p-)(e- + p+) / ((e+ - e-)(e+ + p-)) (Eq. (17) of
    arXiv:2607.27874) with a sonic front, v+ = -c_s,out(e+), and the inside
    state behind the wall at its turning point, e- = e*: the slowest inflow
    that still has a weak inside state behind it.  So G(e+) >= 0 where a
    sonic front at e+ has a weak inside state behind it, which inverse
    hybrids need, and the roots of G are the outside energies of the
    limiting hybrids, whose wall state sits at the inside turning point.

    Assumes that the weak inside states start at the turning point (c_s,in
    vanishes there and the branch has no interior Jouguet point), as for the
    analytic model and the holographic tables.
    """
    e = np.asarray(e, dtype=float)
    e_star = float(inside.e_turn)
    p1 = float(inside.p(e_star))
    p = np.asarray(outside.p(e), dtype=float)
    cs2 = np.asarray(outside.cs2(e), dtype=float)
    return cs2 * (e - e_star) * (e + p1) - (p - p1) * (e_star + p)


def inverse_hyb_obstruction_energies(outside, inside, n_grid=20001):
    """
    Where the wall obstruction lets inverse hybrids exist, searched on the
    outside branch above the inside turning point (the sonic front state
    lies above it on the branch of weak states).  Needs only the equation of
    state, so it tells beforehand whether an EoS can have inverse hybrids.

    Returns
    -------
    roots     : ndarray          Outside energies e+* of the limiting inverse
                                 hybrids, where ``inverse_hyb_wall_obstruction``
                                 vanishes.
    intervals : list of (lo, hi) Stretches of e+ where it is >= 0; empty if the
                                 equation of state rules out inverse hybrids.
    """
    lo = max(float(outside.e_lo), float(inside.e_turn))
    hi = float(outside.e_hi)
    if not hi > lo:
        return np.array([]), []
    e = np.linspace(lo, hi, n_grid)[1:-1]
    G = inverse_hyb_wall_obstruction(e, outside, inside)
    fin = np.isfinite(G)
    g = lambda x: float(inverse_hyb_wall_obstruction(x, outside, inside))
    cells = np.where(fin[:-1] & fin[1:] & (G[:-1] * G[1:] < 0))[0]
    roots = {int(i): brentq(g, e[i], e[i + 1], xtol=1e-12) for i in cells}

    # runs of G >= 0, closed by the roots at their ends (or by the grid)
    pos = fin & (G >= 0)
    edges = np.flatnonzero(np.diff(np.r_[0, pos.astype(int), 0]))
    intervals = [(roots.get(a - 1, float(e[a])), roots.get(b - 1, float(e[b - 1])))
                 for a, b in zip(edges[::2], edges[1::2])]
    return np.array(sorted(roots.values())), intervals


def spinodal_inverse_hyb_contour_finder(xiw, ep, outside, inside):
    """
    Limiting inverse hybrid with wall velocity ``xiw`` and sonic front at the
    outside energy ``ep``, a root of ``inverse_hyb_wall_obstruction``: the
    inside state behind the wall sits at its turning point e*,
        vm = vp (ep + p1) / (e* + p+),   vp = -c_s,out(ep).
    Beyond the root no sonic front has a weak inside state behind it, so
    these bound the inverse hybrids, as the slowest inverse detonations
    (``spinodal_inverse_det_contour_finder``) bound the inverse detonations.
    The front is fixed by the root, so the curve is swept in ``xiw``.

    Parameters
    ----------
    xiw     : float   Wall velocity.
    ep      : float   Outside energy at the sonic front (a root e+*).
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).

    Returns
    -------
    (eN, eC) : (float, float) or (NaN, NaN)
        NaN where that hybrid cannot be built: the wall is not slower than
        the sonic front (xiw >= c_s,out(ep)), the rarefaction in front folds
        (above the degenerate critical point of the outside phase), or no
        shock fits behind the wall (beyond the inverse Jouguet curve).
    """
    xiw = float(xiw.item() if np.ndim(xiw) > 0 else xiw)
    ep = float(ep)
    em = float(inside.e_turn)
    pm, pp = float(inside.p(em)), float(outside.p(ep))
    cs2 = float(outside.cs2(ep))
    if not (np.isfinite(cs2) and cs2 > xiw**2):
        return np.nan, np.nan
    vp = -float(np.sqrt(cs2))
    vm = vp * (ep + pm) / (em + pp)
    if not -1.0 < vm < -xiw:
        return np.nan, np.nan
    eN, eC, _ = find_inverse_hyb(ep, xiw, outside, inside, wall=(vm, em))
    if not (np.isfinite(eN) and np.isfinite(eC)):
        return np.nan, np.nan
    return float(eN), float(eC)


def limiting_inverse_hyb_contour_finder(xi0, outside, inside, e_star=None, v_start=1e-8, t_max=1e7,
                                        xi_min=1e-6, n_grid=20000, n_samples=2000, n_side=200):
    """
    Limiting inverse hybrid whose rarefaction in front of the wall reaches
    rest at xi0 = c_s(E_N) with dv/dxi -> infinity: the flow obstruction of
    the second kind, as for the limiting inverse deflagrations
    (``limiting_inverse_def_contour_finder``), with the same start.

    Above the DCP, E_N > e*, the rarefaction turns back before reaching rest;
    the limiting one has its fold at the point of rest, of size ``v_start``.
    It is started there and integrated towards the centre: where the
    junction conditions with the centre at rest hold it would meet the wall
    of the limiting inverse deflagration, and further in, where the flow
    becomes sonic, mu = c_s(e), it meets the wall of the limiting hybrid,
    whose front state is then sonic, v+ = -c_s(e+).  Behind the wall come
    the weak inside state, the compression and the shock to rest, as in
    ``find_inverse_hyb``.  Sweeping xi0 in (0, c_s(e*)) gives the separator
    that bounds the inverse hybrids from above in E_N; it ends at
    xi_w = c_s(e*), E_N = e+ = e*, where the rarefaction shrinks to nothing.
    The curve moves (logarithmically) with ``v_start``, whose default equals
    the ``v_rest`` of ``find_inverse_hyb``.

    Parameters
    ----------
    xi0       : float   Point of rest, xi0 = c_s(E_N); below c_s(e*).
    outside   : Branch  Metastable phase the bubble expands into (LT).
    inside    : Branch  Phase inside the bubble (HT).
    e_star    : float   DCP of the outside branch (computed if None).
    v_start   : float   Size of the fold at the point of rest.
    t_max     : float   Maximum integration length of the rarefaction.
    xi_min    : float   The rarefaction is followed down to xi_min.
    n_grid    : int     Points of the scan for the weak inside state.
    n_samples : int     Points of the flow behind the wall searched for the shock.
    n_side    : int     Points of the regions at rest in the returned flow.

    Returns
    -------
    (xiw, eN, eC, ep, flow) : the wall velocity, nucleation and centre
        energies, the outside energy at the sonic front, and [v(xi), xi, e(xi)]
        from the centre to xi = 1; or NaNs where that hybrid cannot be built
        (no sonic point, no weak inside state behind it, no shock to rest).
    """
    fail = (np.nan,) * 5
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

    # rarefaction towards the centre, until it becomes sonic again: the wall
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
    sonic = lambda t: (lambda v, x, e: (x - v)**2 - float(outside.cs2(e)) * (1.0 - x * v)**2)(*sol.sol(t))
    V, XI, EE = sol.sol(ts)
    S = (XI - V)**2 - np.asarray(outside.cs2(EE), dtype=float) * (1.0 - XI * V)**2
    # leave out the start, which is on the sonic curve (the fold at the point of rest)
    inside_flow = np.where(S < 0)[0]
    if not inside_flow.size:
        return fail
    k0 = inside_flow[0]
    cross = k0 + np.where((S[k0:-1] < 0) & (S[k0 + 1:] >= 0))[0]
    if not cross.size:
        return fail
    k = cross[0]
    t_w = brentq(sonic, ts[k], ts[k + 1], xtol=1e-14)
    vw, xw, ew = sol.sol(t_w)
    vp = (vw - xw) / (1.0 - vw * xw)                     # = -c_s(ew) on the sonic curve
    if not (0.0 < xw < 1.0 and vp < 0.0):
        return fail
    v_r, xi_r, e_r = (np.append(a[:k + 1], b) for a, b in zip((V, XI, EE), (vw, xw, ew)))
    if np.any(np.diff(xi_r) > 1e-12 * xi0) or np.any(v_r > 0):
        return fail

    # wall: the weak inside state behind the sonic front; then the shock to rest
    states = _weak_inside_states(vp, float(ew), float(xw), outside, inside, n_grid)
    if not states:
        return fail
    vm, em = states[0]
    behind = _shock_to_rest_behind(inside, float(xw), vm, em, 1e-8, 1e3, n_samples)
    if behind is None:
        return fail
    eC, xi_sh, v_f, xi_f, e_f = behind
    flow = [np.concatenate((np.zeros(n_side), v_f[::-1], v_r[::-1], np.zeros(n_side))),
            np.concatenate((np.linspace(0.0, xi_sh, n_side), xi_f[::-1], xi_r[::-1], np.linspace(xi0, 1.0, n_side))),
            np.concatenate((np.full(n_side, eC), e_f[::-1], e_r[::-1], np.full(n_side, eN)))]
    return float(xw), float(eN), float(eC), float(ew), flow
