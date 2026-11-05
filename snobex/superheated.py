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
* **Inverse detonation** – ``find_inverse_det``: supersonic wall with the
  outside at rest, the inside moving inwards behind it and brought to rest
  by a compression shock; scanned over the (xi_w, E_N) plane by
  ``scan_inverse_detonations``.

``superheated_window`` gives T_c and the energies at which the outside phase
is superheated; ``inverse_def_entropy_jump`` and
``inverse_det_entropy_jump`` check the entropy condition at the wall;
``alpha_n`` is the transition strength alpha_N.
"""

import numpy as np
from scipy.optimize import brentq

from .hydrodynamics import integrate_flow, outside_state_at_wall, rest_behind_residual
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


def find_inverse_det(eN, xiw, outside, inside, v_rest=1e-8, t_max=1e3, n_grid=20000, n_samples=2000,
                     wall=None):
    """
    Inverse detonation of a bubble expanding at ``xiw`` into the outside
    phase at rest with nucleation energy ``eN``: the inverse deflagration
    pipeline with the nucleation energy as input, as the shock of a
    deflagration is placed with the centre energy as input.

    Assumed structure, from the centre outwards:
      * 0 < xi < xi_sh: inside phase at rest, e = eC.
      * xi_sh: compression shock moving outwards, with the fluid behind it
        (towards the centre) at rest.
      * xi_sh < xi < xiw: inside phase moving inwards (v < 0), the
        self-similar flow from the wall state.
      * Wall at xiw, supersonic with respect to the fluid ahead,
        xiw > c_s,out(eN).  The inside state (vm, em) follows from the
        junction conditions with v+ = -xiw, e+ = eN, taking the weak root:
        |vm| > xiw (the fluid behind the wall moves inwards) and
        |vm| > c_s,in(em) (supersonic behind the wall).
      * xi > xiw: outside phase at rest, e = eN.

    The flow behind the wall slows down inwards, but in a convex inside phase
    it reaches the sonic curve and turns back before coming to rest.  The
    shock is placed on the stretch before that fold, where the junction
    conditions with the fluid at rest behind it hold
    (``rest_behind_residual``), and must satisfy the Lax conditions:
    supersonic inflow, |u1| > c_s(e1), and subsonic outflow, xi_sh < c_s(eC).
    If the flow comes down to |v| < ``v_rest`` before folding, it joins the
    fluid at rest smoothly (the shock has vanished; as for the folds of
    ``find_inverse_def``, smaller ones count as rest), and eC is the energy
    there.

    Parameters
    ----------
    eN        : float   Nucleation energy (outside branch).
    xiw       : float   Wall velocity.
    outside   : Branch  Metastable phase the bubble expands into (LT).
    inside    : Branch  Phase inside the bubble (HT).
    v_rest    : float   |v| below which the flow behind the wall counts as at rest.
    t_max     : float   Maximum integration length in the flow parameter t.
    n_grid    : int     Points of the scan for the wall state on the inside branch.
    n_samples : int     Points of the flow behind the wall searched for the shock.
    wall      : (vm, em), optional
        Inside state behind the wall to use instead of solving the junction
        conditions, e.g. the state at the inside turning point of the
        slowest inverse detonation.

    Returns
    -------
    eC : float or NaN
        Energy at the centre of the bubble, behind the shock.
    flow : list [v(xi), xi, e(xi)] or NaN
        Profile from the centre to xi = 1.
    """
    if np.ndim(eN) > 0:
        eN = eN.item()
    if not (outside.e_lo <= eN <= outside.e_hi) or not 0.0 < xiw < 1.0:
        return np.nan, np.nan
    cs2_out = float(outside.cs2(eN))
    if not np.isfinite(cs2_out) or xiw**2 <= cs2_out:
        return np.nan, np.nan

    # ------------------------------------------------------------------
    # (1) Wall: the weak inside state, from the junction conditions reduced
    #     to one equation in em (v- = v+ (e+ + p-) / (e- + p+))
    # ------------------------------------------------------------------
    if wall is None:
        vp, pp = -xiw, float(outside.p(eN))
        top = inside.e_hi if np.isfinite(inside.e_hi) else 40.0 * max(inside.e_turn, 1.0)
        # geometric from the turning point: the wall states of the slowest
        # walls lie just above it (see spinodal_inverse_det_contour_finder)
        grid = inside.e_lo + (top - inside.e_lo) * np.geomspace(1e-9, 1.0, n_grid)[:-1]

        def jc(em):
            pm = np.asarray(inside.p(em), dtype=float)
            return vp**2 * (eN + pm) * (eN - em) - (pp - pm) * (em + pp)

        J = jc(grid)
        for i in np.where(np.isfinite(J[:-1]) & np.isfinite(J[1:]) & (J[:-1] * J[1:] < 0))[0]:
            em = brentq(lambda x: float(jc(x)), grid[i], grid[i + 1], xtol=1e-14)
            vm = vp * (eN + float(inside.p(em))) / (em + pp)
            if -1.0 < vm < -xiw and vm**2 > float(inside.cs2(em)):
                wall = (vm, em)
                break
        if wall is None:
            return np.nan, np.nan
    vm, em = wall

    # ------------------------------------------------------------------
    # (2) Flow behind the wall, inwards (supersonic: xi decreases with t),
    #     up to the fold where it turns back
    # ------------------------------------------------------------------
    def fold(t, y, *args):
        v, xi, e = y
        return (xi - v)**2 - float(inside.cs2(e)) * (1.0 - xi * v)**2
    fold.terminal = True

    def at_rest(t, y, *args):
        return y[0] + v_rest
    at_rest.terminal = True
    at_rest.direction = 1

    sol = integrate_flow(inside, [(vm + xiw) / (1.0 + xiw * vm), xiw, em], (0.0, -t_max),
                         events=[fold, at_rest], rtol=1e-10, atol=[1e-4 * v_rest, 1e-12, 1e-12],
                         dense_output=True)
    ts = np.linspace(0.0, sol.t[-1], n_samples)

    # ------------------------------------------------------------------
    # (3) Compression shock to rest, the first one inwards from the wall;
    #     near rest the residual is round-off, so that tail is left out
    # ------------------------------------------------------------------
    resid = lambda t: rest_behind_residual(*(lambda y: (y[1], y[0], y[2]))(sol.sol(t)), inside, inside)[0]
    R = np.array([resid(t) for t in ts])
    R[np.abs(sol.sol(ts)[0]) < 2.0 * v_rest] = np.nan
    cross = np.where(np.isfinite(R[:-1]) & np.isfinite(R[1:]) & (R[:-1] * R[1:] < 0))[0]
    if cross.size:
        k = cross[0]
        t_sh = brentq(resid, ts[k], ts[k + 1], xtol=1e-12)
        v_sh, xi_sh, e_sh = sol.sol(t_sh)
        _, eC = rest_behind_residual(xi_sh, v_sh, e_sh, inside, inside)
        u1 = (v_sh - xi_sh) / (1.0 - v_sh * xi_sh)
        if not (u1**2 > float(inside.cs2(e_sh)) and xi_sh**2 < float(inside.cs2(eC))):
            return np.nan, np.nan
        t_f = np.append(ts[:k + 1], t_sh)
    elif sol.t_events[-1].size:
        # no shock before the flow comes to rest: it joins the centre smoothly
        t_f = ts
        xi_sh, eC = float(sol.y[1][-1]), float(sol.y[2][-1])
    else:
        return np.nan, np.nan

    v_f, xi_f, e_f = sol.sol(t_f)                               # from the wall to the shock
    xi = np.concatenate((np.linspace(0, xi_sh, 1000), xi_f[::-1], np.linspace(xiw, 1, 1000)))
    v = np.concatenate((np.zeros(1000), v_f[::-1], np.zeros(1000)))
    e = np.concatenate((eC * np.ones(1000), e_f[::-1], eN * np.ones(1000)))
    return float(eC), [v, xi, e]


def _entropy_current(s, v):
    """Entropy current s gamma v through the wall, with v the signed wall-frame velocity."""
    return s * v / np.sqrt(1.0 - v**2)


def inverse_def_entropy_jump(eC, xiw, flow, outside, inside):
    """
    Entropy condition at the wall of an inverse deflagration (a ``flow``
    returned by ``find_inverse_def``), in the form of ``entropy_checker``:
        Delta J_s = J_s+ - J_s-,   J_s = s gamma v,
    with v the wall-frame velocity of the outside state just in front of the
    wall (+) and of the inside one at rest behind it (-).  Both move
    inwards (v < 0), so Delta J_s >= 0 means that at least as much entropy
    leaves the wall into the bubble as enters it.  NaN if a branch has no
    entropy.
    """
    if outside.s is None or inside.s is None:
        return np.nan
    v, xi, e = flow
    i = int(np.argmax(v < 0))                  # first point in front of the wall
    vp = (v[i] - xiw) / (1.0 - v[i] * xiw)
    return _entropy_current(float(outside.s(e[i])), vp) - _entropy_current(float(inside.s(eC)), -xiw)


def inverse_det_entropy_jump(eN, xiw, flow, outside, inside):
    """
    Entropy condition Delta J_s = J_s+ - J_s- >= 0 at the wall of an inverse
    detonation (a ``flow`` returned by ``find_inverse_det``), as
    ``inverse_def_entropy_jump``: in front of the wall the outside phase at
    rest, behind it the last state of the inside flow, moving inwards (the
    shock behind it, compressive in a convex phase, raises the entropy as
    well).  NaN if a branch has no entropy.
    """
    if outside.s is None or inside.s is None:
        return np.nan
    v, xi, e = flow
    i = int(np.where(v < 0)[0][-1])            # last point behind the wall
    vm = (v[i] - xiw) / (1.0 - v[i] * xiw)
    return _entropy_current(float(outside.s(eN)), -xiw) - _entropy_current(float(inside.s(e[i])), vm)


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
        alpha_N = 4 (theta_H - theta_L) / (3 w_out),
    with theta = (e - 3p) / 4 of the HT and LT states at T_N and w_out = e + p
    of the metastable phase, the outside one.  For a superheated transition
    that is w_L, as in Eq. (23) of Bea, Casalderrey-Solana, Mateos and
    Sanchez-Garitaonandia, arXiv:2406.14450; for a supercooled one w_H, the
    convention of Mateos et al. used by ``compute_alphan``.  It is positive in
    both cases.  NaN if a branch has no temperature or no inside state is at
    that temperature.
    """
    if outside.T is None or inside.T is None or not np.isfinite(eN):
        return np.nan
    e_in = _energy_at_temperature(inside, float(outside.T(eN)), _branch_grid(inside, 4000))
    if e_in is None:
        return np.nan
    out, ins = (float(eN), float(outside.p(eN))), (e_in, float(inside.p(e_in)))
    (e_H, p_H), (e_L, p_L) = (out, ins) if outside.up else (ins, out)
    return 4.0 / 3.0 * (0.25 * (e_H - 3.0 * p_H) - 0.25 * (e_L - 3.0 * p_L)) / (out[0] + out[1])


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


def scan_inverse_detonations(outside, inside, xiw_range, en_range, n_xiw=25, n_en=25,
                             n_coarse=24, progress=None):
    """
    Inverse detonations on a grid of wall velocities and nucleation energies,
    as ``scan_inverse_deflagrations`` with the nucleation energy, the input of
    ``find_inverse_det``, as the (evenly spaced) row variable: a coarse
    n_coarse x n_coarse pass locates where solutions exist and the
    n_xiw x n_en grid is laid over that region (``n_coarse=0`` grids the
    given ranges directly).

    Parameters
    ----------
    outside, inside : Branch  Metastable outside phase (LT) and bubble phase (HT).
    xiw_range       : (lo, hi) Wall velocities.
    en_range        : (lo, hi) Nucleation energies on the outside branch.
    progress        : callable(done, total), optional, called after each xi_w
                      of both passes.

    Returns
    -------
    list of (xiw, eN, eC, flow)
    """
    total = n_coarse + n_xiw

    def solve(xiws, eNs, done_before):
        rows = []
        for k, xiw in enumerate(xiws):
            for eN in eNs:
                eC, flow = find_inverse_det(eN, xiw, outside, inside)
                if np.isfinite(eC):
                    rows.append((xiw, eN, eC, flow))
            if progress is not None:
                progress(done_before + k + 1, total)
        return rows

    if n_coarse:
        cx, ce = np.linspace(*xiw_range, n_coarse), np.linspace(*en_range, n_coarse)
        coarse = solve(cx, ce, 0)
        if not coarse:
            return []
        # pad the region by two coarse steps: its lower edge in E_N slants across the grid
        x, en = np.array([r[0] for r in coarse]), np.array([r[1] for r in coarse])
        dx = cx[1] - cx[0]
        i_lo, i_hi = np.searchsorted(ce, en.min()), np.searchsorted(ce, en.max())
        xiw_range = (max(xiw_range[0], x.min() - 2 * dx), min(xiw_range[1], x.max() + 2 * dx))
        en_range = (ce[max(i_lo - 2, 0)], ce[min(i_hi + 2, len(ce) - 1)])
    return solve(np.linspace(*xiw_range, n_xiw), np.linspace(*en_range, n_en), n_coarse)
