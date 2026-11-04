"""
snobex.hydrodynamics
=====================
Hydrodynamics shared by supercooled and superheated transitions.

* ``system_minus``: the self-similar relativistic fluid equations, valid on
  either branch (the name is historical).
* ``_edge_events``: stops a flow at the edge of a tabulated branch.
* Building blocks on ``Branch`` objects for the phases outside and inside
  the bubble: ``junction_residual``, ``integrate_flow``,
  ``inside_state_at_wall``, ``outside_state_at_wall`` and
  ``pressure_balance_energy``.

The solutions themselves live in ``supercooled`` and ``superheated``.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq


_DEF_RESIDUAL_TOL = 1e-4


# ---------------------------------------------------------------------------
# Relativistic fluid ODE (self-similar frame)
# ---------------------------------------------------------------------------

def system_minus(t, y, p_func, cs2_func, eT, p0):
    """
    Self-similar relativistic fluid equations in the variable t = 2/(xi·v·(xi−v)).

    State vector y = [v, xi, e].

    Parameters
    ----------
    t        : float     Integration variable.
    y        : array     [v, xi, e].
    p_func   : callable  Pressure function p(e).
    cs2_func : callable  Speed-of-sound squared c_s²(e).
    eT       : float     Unused turning-point argument (kept for API consistency).
    p0       : float     Unused pressure offset argument (kept for API consistency).

    Returns
    -------
    list  [dv/dt, dxi/dt, de/dt]
    """
    v, xi, e = y
    cs2 = cs2_func(e)
    dv_dt  = 2.0 * v * cs2 * (1.0 - v**2) * (1.0 - xi * v)
    dxi_dt = xi * ((xi - v)**2 - cs2 * (1.0 - xi * v)**2)
    de_dt  = 2.0 * (e + p_func(e, eT, p0)) * v * (xi - v)
    return [dv_dt, dxi_dt, de_dt]


def _edge_events(cs2_func):
    """
    Terminal events for ``solve_ivp(system_minus, ...)`` that stop a flow
    just before it leaves the energy range of a tabulated EoS branch, whose
    interpolants return NaN outside it (see ``load_custom_eos``).  Without
    them the integrator keeps shrinking its step towards the edge of the
    table and never finishes.  Analytic EoS callables extrapolate and get
    no events, so their flows are unchanged.
    """
    if getattr(cs2_func, 'extrapolate', True):
        return None
    lo, hi = float(cs2_func.x[0]), float(cs2_func.x[-1])
    tol = 1e-6 * (hi - lo)

    def _below(t, y, *args):
        return y[2] - (lo + tol)

    def _above(t, y, *args):
        return (hi - tol) - y[2]

    for ev in (_below, _above):
        ev.terminal = True
        ev.direction = -1   # only when moving out of the table
    return [_below, _above]


# ---------------------------------------------------------------------------
# Junction-condition residuals
# ---------------------------------------------------------------------------

def _jc_residual(vp, vm, ep, em, pp_val, pm_val):
    """Core 2×1 junction-condition vector (energy–momentum conservation)."""
    JC1 = vp * vm * (ep - em) - (pp_val - pm_val)
    JC2 = vp * (ep + pm_val) - (em + pp_val) * vm
    if np.any(np.isnan([JC1, JC2])):
        return np.array([1e10, 1e10])
    return np.array([JC1, JC2])


# ---------------------------------------------------------------------------
# Building blocks on Branch objects (outside / inside phase)
# ---------------------------------------------------------------------------

def junction_residual(vp, ep, vm, em, outside, inside):
    """Energy–momentum conservation across a wall between the ``outside``
    state (vp, ep) and the ``inside`` state (vm, em), in the wall frame."""
    return _jc_residual(vp, vm, ep, em, outside.p(ep), inside.p(em))


def integrate_flow(branch, y0, t_span, t_eval=None, events=(), **kwargs):
    """
    Integrate the self-similar flow equations on ``branch`` from
    y0 = [v, xi, e] over ``t_span``, stopping at the edge of a tabulated
    branch.  ``events`` are further ``solve_ivp`` events, checked after the
    edge stop; other keyword arguments are passed to ``solve_ivp``.
    """
    all_events = list(_edge_events(branch.cs2) or []) + list(events)
    return solve_ivp(system_minus, t_span, y0, t_eval=t_eval,
                     args=(branch.pressure, branch.cs2, branch.e_turn, None),
                     events=all_events or None, **kwargs)


def inside_state_at_wall(outside, inside, e_out, vp, guesses, v_bounds=(-1.0, 0.0)):
    """
    State (vm, em) on the ``inside`` branch just behind a wall whose outside
    state is (vp, e_out), or None if the junction conditions have no
    solution on that branch.  Each guess (vm, em) seeds ``least_squares``.
    """
    sol_m = res = None
    for guess in guesses:
        try:
            sol = least_squares(
                lambda x: junction_residual(vp, e_out, x[0], x[1], outside, inside),
                guess, bounds=([v_bounds[0], inside.e_lo], [v_bounds[1], inside.e_hi]),
                ftol=1e-12, xtol=1e-12)
        except Exception:
            continue
        sol_m = sol.x
        res = junction_residual(vp, e_out, sol.x[0], sol.x[1], outside, inside)
        if (not np.any(np.isnan(res))) and np.mean(np.array(res)**2) < 1e-8:
            break

    if sol_m is None or np.isnan(sol_m).any() or np.mean(np.array(res)**2) > 1e-10:
        return None
    return sol_m


def outside_state_at_wall(outside, inside, e_in, vm, guesses, v_bounds,
                          tol=_DEF_RESIDUAL_TOL):
    """
    State (vp, ep) on the ``outside`` branch just in front of a wall whose
    inside state is (vm, e_in), or None.  Each guess (vp, ep) seeds
    ``least_squares``; the first converged solution on the branch with
    vp != 0 is returned.
    """
    for guess in guesses:
        sol = least_squares(
            lambda x: junction_residual(x[0], x[1], vm, e_in, outside, inside),
            guess, bounds=([v_bounds[0], outside.e_lo], [v_bounds[1], outside.e_hi]))
        vp, ep = sol.x
        res = junction_residual(vp, ep, vm, e_in, outside, inside)
        if not (np.any(np.isnan(res)) or np.mean(np.array(res)**2) > tol
                or ep < outside.e_lo or ep > outside.e_hi or abs(vp) < 1e-15):
            return vp, ep
    return None


def pressure_balance_energy(outside, inside):
    """
    Energy on the ``outside`` branch whose pressure equals that of the
    ``inside`` branch at its turning point, searched from the outside
    turning point into the branch.  NaN if the outside pressure is undefined.
    """
    target = inside.p(inside.e_turn)
    if not np.isfinite(target):
        return np.nan

    e0 = float(outside.e_turn)
    if outside.up:
        e_scan = np.linspace(max(e0, 1e-10),
                             max(e0 * 20.0, float(inside.e_turn) * 20.0, e0 + 10.0), 2000)
        clamp = lambda e: max(e, e0 + 1e-9)
    else:
        e_scan = np.linspace(e0, float(outside.e_lo), 2000)
        clamp = lambda e: min(e, e0 - 1e-9)
    p_scan = np.asarray(outside.p(e_scan), dtype=float)
    finite = np.isfinite(p_scan)

    if not np.any(finite):
        return np.nan

    ef = e_scan[finite]
    ff = target - p_scan[finite]
    sign_change = np.where(np.sign(ff[:-1]) * np.sign(ff[1:]) <= 0)[0]

    if sign_change.size > 0:
        i = int(sign_change[0])
        a, b = sorted((float(ef[i]), float(ef[i + 1])))
        try:
            return clamp(float(brentq(lambda e: target - float(outside.p(e)), a, b)))
        except Exception:
            pass

    i_best = int(np.nanargmin(np.abs(ff)))
    return clamp(float(ef[i_best]))
