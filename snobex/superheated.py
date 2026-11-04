"""
snobex.superheated
==================
Self-similar bubble solutions of a superheated transition: a bubble of the
HT phase (inside) expanding into the metastable LT phase (outside, at rest
with nucleation energy eN).

Every finder takes the two phases as ``Branch`` objects, ``outside`` (LT)
and ``inside`` (HT), and is built on the building blocks of
``snobex.hydrodynamics``.  Solution types implemented so far:

* **Inverse deflagration** – ``find_inverse_def``, scanned over the
  (xi_w, E_C) plane by ``scan_inverse_deflagrations``.

``superheated_window`` gives T_c and the energies at which the outside phase
is superheated; ``inverse_def_entropy_ratio`` checks the entropy condition
at the wall; ``alpha_n`` is the transition strength alpha_N.
"""

import numpy as np
from scipy.optimize import brentq

from .hydrodynamics import integrate_flow, outside_state_at_wall
from .thermodynamics import find_critical_temperature


def find_inverse_def(eC, xiw, outside, inside, v_rest=1e-8, t_max=1e4, tol=1e-10, wall=None):
    """
    Inverse deflagration of a bubble whose interior is at rest with energy
    ``eC`` and whose wall moves at ``xiw``.

    Assumed structure:
      * 0 < xi < xiw: inside phase at rest, e = eC.
      * Wall at xiw: the outside state (vp, ep) follows from the junction
        conditions with |vp| > xiw, so the fluid just in front of the wall
        moves inwards (v < 0).  The flow is subsonic on both sides:
        xiw <= c_s(eC) and |vp| < c_s(ep).
      * xiw < xi < xi_r: a rarefaction in the outside phase, v < 0, along
        which v rises to 0 at the sound cone xi_r = c_s(eN).  No shock.
      * xi > xi_r: outside phase at rest with e = eN.

    Parameters
    ----------
    eC      : float   Energy at the centre of the bubble (inside branch).
    xiw     : float   Wall velocity.
    outside : Branch  Metastable phase the bubble expands into (LT).
    inside  : Branch  Phase inside the bubble (HT).
    v_rest  : float   |v| below which the rarefaction counts as at rest.
    t_max   : float   Maximum integration length in the flow parameter t.
    tol     : float   Acceptance on the mean squared junction residual.
    wall    : (vp, ep), optional
        Outside state at the wall to use instead of solving the junction
        conditions, e.g. the sonic state of an inverse Jouguet deflagration.

    Returns
    -------
    eN : float or NaN
        Nucleation energy (outside phase at rest).
    flow : list [v(xi), xi, e(xi)] or NaN
        Profile from the centre to xi = 1.
    """
    if np.ndim(eC) > 0:
        eC = eC.item()
    if not (inside.e_lo <= eC <= inside.e_hi):
        return np.nan, np.nan
    cs2_in = float(inside.cs2(eC))
    if not np.isfinite(cs2_in) or cs2_in < xiw**2:
        return np.nan, np.nan

    # ------------------------------------------------------------------
    # (1) Wall: outside state with the fluid moving inwards, |vp| > xiw,
    #     unless one is given
    # ------------------------------------------------------------------
    vm = -xiw
    e_lo = outside.e_lo
    e_hi = outside.e_hi if np.isfinite(outside.e_hi) else 10.0 * max(outside.e_turn, 1.0)
    for f_v in ((0.1, 0.5) if wall is None else ()):
        for f_e in (0.5, 0.25, 0.75, 0.9):
            guess = (-(xiw + f_v * (1.0 - xiw)), e_lo + f_e * (e_hi - e_lo))
            state = outside_state_at_wall(outside, inside, eC, vm, guesses=[guess],
                                          v_bounds=(-1.0, -xiw), tol=tol)
            if state is None:
                continue
            cs2_out = float(outside.cs2(state[1]))
            if np.isfinite(cs2_out) and state[0]**2 < cs2_out:
                wall = state
                break
        if wall is not None:
            break
    if wall is None:
        return np.nan, np.nan
    vp, ep = wall

    # ------------------------------------------------------------------
    # (2) Rarefaction in front of the wall, integrated until v ~ 0.
    #     Subsonic flow means xi grows as t decreases.
    # ------------------------------------------------------------------
    v0 = (vp + xiw) / (1.0 + xiw * vp)

    def at_rest(t, y, *args):
        return y[0] + v_rest
    at_rest.terminal = True
    at_rest.direction = 1

    sol = integrate_flow(outside, [v0, xiw, ep], (0.0, -t_max), events=[at_rest],
                         rtol=1e-10, atol=1e-12)
    if sol.t_events[-1].size == 0:
        return np.nan, np.nan
    v_r, xi_r, e_r = sol.y
    if np.any(np.diff(xi_r) < 0) or np.any(v_r > 0):
        return np.nan, np.nan
    eN = float(e_r[-1])
    if not (outside.e_lo <= eN <= outside.e_hi):
        return np.nan, np.nan

    xi = np.concatenate((np.linspace(0, xiw, 1000), xi_r, np.linspace(xi_r[-1], 1, 1000)))
    v = np.concatenate((np.zeros(1000), v_r, np.zeros(1000)))
    e = np.concatenate((eC * np.ones(1000), e_r, eN * np.ones(1000)))
    return eN, [v, xi, e]


def inverse_def_entropy_ratio(eC, xiw, flow, outside, inside):
    """
    Entropy flux leaving the wall into the bubble over the flux entering it
    from the outside phase, |s gamma v|_inside / |s gamma v|_outside in the
    wall frame, for a ``flow`` returned by ``find_inverse_def``.  The entropy
    condition requires a ratio >= 1.  NaN if a branch has no entropy.
    """
    if outside.s is None or inside.s is None:
        return np.nan
    v, xi, e = flow
    i = int(np.argmax(v < 0))                  # first point in front of the wall
    vp = (v[i] - xiw) / (1.0 - v[i] * xiw)
    gamma = lambda u: 1.0 / np.sqrt(1.0 - u**2)
    return (float(inside.s(eC)) * gamma(xiw) * xiw) / (float(outside.s(e[i])) * gamma(vp) * abs(vp))


# ---------------------------------------------------------------------------
# Thermodynamic window and scan
# ---------------------------------------------------------------------------

def _branch_grid(branch, n):
    """Energies along ``branch`` from its far end to its turning point."""
    if branch.up:
        top = branch.e_hi if np.isfinite(branch.e_hi) else 10.0 * branch.e_turn
        return np.linspace(branch.e_turn, top, n)
    return np.linspace(branch.e_lo + 1e-3 * (branch.e_turn - branch.e_lo), branch.e_turn, n)


def _energy_at_temperature(branch, T_target, grid):
    """First energy on ``grid`` where the branch temperature equals ``T_target``."""
    dT = np.asarray(branch.T(grid), float) - T_target
    ok = np.isfinite(dT)
    g, dT = grid[ok], dT[ok]
    cross = np.where(np.sign(dT[:-1]) != np.sign(dT[1:]))[0]
    if len(cross) == 0:
        return None
    i = cross[0]
    return float(brentq(lambda x: float(branch.T(x)) - T_target, g[i], g[i + 1]))


def superheated_window(outside, inside):
    """
    Critical temperature and the energies of both phases at T_c.  The outside
    phase is superheated (metastable) for E_N in [e_out(T_c), outside.e_turn].

    Returns
    -------
    (Tc, e_out_Tc, e_in_Tc) : floats, None where they cannot be found
        (e.g. a branch without temperature).
    """
    if outside.T is None or inside.T is None:
        return None, None, None
    g_out, g_in = _branch_grid(outside, 2000), _branch_grid(inside, 4000)
    T_out, T_in = np.asarray(outside.T(g_out), float), np.asarray(inside.T(g_in), float)
    ok_o, ok_i = np.isfinite(T_out), np.isfinite(T_in)
    Tc = find_critical_temperature(T_out[ok_o], -np.asarray(outside.p(g_out), float)[ok_o],
                                   T_in[ok_i], -np.asarray(inside.p(g_in), float)[ok_i])
    if Tc is None:
        return None, None, None
    return Tc, _energy_at_temperature(outside, Tc, g_out), _energy_at_temperature(inside, Tc, g_in)


def alpha_n(eN, outside, inside):
    """
    Transition strength at the nucleation temperature T_N = T_out(eN),
        alpha_N = 4 (theta_H - theta_L) / (3 w_H),
    with theta = (e - 3p) / 4 and w = e + p of the HT and LT states at T_N,
    whichever of them is metastable (the convention of Mateos et al., also
    used by ``compute_alphan``).  It is positive for supercooled and
    superheated transitions alike.  NaN if a branch has no temperature or no
    inside state is at that temperature.
    """
    if outside.T is None or inside.T is None or not np.isfinite(eN):
        return np.nan
    e_in = _energy_at_temperature(inside, float(outside.T(eN)), _branch_grid(inside, 4000))
    if e_in is None:
        return np.nan
    out, ins = (float(eN), float(outside.p(eN))), (e_in, float(inside.p(e_in)))
    (e_H, p_H), (e_L, p_L) = (out, ins) if outside.up else (ins, out)
    return 4.0 / 3.0 * (0.25 * (e_H - 3.0 * p_H) - 0.25 * (e_L - 3.0 * p_L)) / (e_H + p_H)


def _ec_grid(outside, inside, ec_range, n):
    """
    ``n`` centre energies in ``ec_range`` whose pressure-balance partners on
    the outside branch, p_out(E_N) = p_in(E_C) (the xi_w -> 0 limit of an
    inverse deflagration), are evenly spaced in E_N.  E_N(E_C) steepens as
    the outside sound speed vanishes towards its turning point, so an even
    E_C grid would leave the top of the E_N range unsampled.  Falls back to
    an even E_C grid if the pressures cannot be matched.
    """
    even = np.linspace(*ec_range, n)
    a, b = float(outside.e_lo), float(outside.e_hi)
    if not np.isfinite(b):
        return even
    p_in = lambda e: float(inside.p(e))
    p_out = lambda e: float(outside.p(e))

    def eN_of(eC):
        P = p_in(eC)
        if not np.isfinite(P):
            return np.nan
        if P <= p_out(a):
            return a
        if P >= p_out(b):
            return b
        return brentq(lambda e: p_out(e) - P, a, b)

    def eC_of(eN):
        P = p_out(eN)
        lo, hi = ec_range
        f_lo, f_hi = p_in(lo) - P, p_in(hi) - P
        if f_lo >= 0:                             # the grid ends map back onto the range ends
            return lo
        if f_hi <= 0:
            return hi
        if not f_lo * f_hi < 0:                   # NaN pressures
            return np.nan
        return brentq(lambda e: p_in(e) - P, lo, hi)

    n_lo, n_hi = eN_of(ec_range[0]), eN_of(ec_range[1])
    n_hi = min(n_hi, b - 1e-3 * (b - a))         # the outside sound speed vanishes at b
    if not (np.isfinite(n_lo) and np.isfinite(n_hi) and n_hi > n_lo):
        return even
    eCs = np.array([eC_of(x) for x in np.linspace(n_lo, n_hi, n)])
    eCs = eCs[np.isfinite(eCs)]
    return eCs if len(eCs) >= n // 2 else even


def scan_inverse_deflagrations(outside, inside, xiw_range, ec_range, n_xiw=25, n_ec=25,
                               n_coarse=24, progress=None):
    """
    Inverse deflagrations on a grid of wall velocities and centre energies.

    A coarse n_coarse x n_coarse pass over the given ranges first locates
    where solutions exist, and the n_xiw x n_ec grid is laid over that region
    (inverse deflagrations occupy a small corner of the plane).  Set
    ``n_coarse=0`` to grid the given ranges directly.  The E_C values are
    spaced so that the nucleation energies they map to cover the E_N range
    evenly (see ``_ec_grid``).

    Parameters
    ----------
    outside, inside : Branch  Metastable outside phase (LT) and bubble phase (HT).
    xiw_range       : (lo, hi) Wall velocities.
    ec_range        : (lo, hi) Centre energies on the inside branch.
    progress        : callable(done, total), optional, called after each xi_w
                      of both passes.

    Returns
    -------
    list of (xiw, eC, eN, flow)
    """
    total = n_coarse + n_xiw

    def solve(xiws, eCs, done_before):
        rows = []
        for k, xiw in enumerate(xiws):
            for eC in eCs:
                eN, flow = find_inverse_def(eC, xiw, outside, inside)
                if np.isfinite(eN):
                    rows.append((xiw, eC, eN, flow))
            if progress is not None:
                progress(done_before + k + 1, total)
        return rows

    if n_coarse:
        cx, ce = np.linspace(*xiw_range, n_coarse), _ec_grid(outside, inside, ec_range, n_coarse)
        coarse = solve(cx, ce, 0)
        if not coarse:
            return []
        x, ec = np.array([r[0] for r in coarse]), np.array([r[1] for r in coarse])
        dx = cx[1] - cx[0]
        i_lo, i_hi = np.searchsorted(ce, ec.min()), np.searchsorted(ce, ec.max())
        xiw_range = (max(xiw_range[0], x.min() - dx), min(xiw_range[1], x.max() + dx))
        ec_range = (ce[max(i_lo - 1, 0)], ce[min(i_hi + 1, len(ce) - 1)])
    return solve(np.linspace(*xiw_range, n_xiw), _ec_grid(outside, inside, ec_range, n_ec), n_coarse)
