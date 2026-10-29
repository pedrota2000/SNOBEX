"""
snobex.eos
===========
Equation-of-state (EoS) models for the two-phase fluid.

Two analytic parametrisations are provided:

* ``pplus``  – high-temperature (HT) branch.
* ``pminus`` – low-temperature (LT) branch.

A helper ``speed_of_sound_squared`` wraps both into a uniform callable
and ``load_custom_eos`` allows replacing the analytic models with
tabulated data read from a file, split into its HT and LT branches.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Analytic EoS parametrisations
# ---------------------------------------------------------------------------

def pplus(e, eHT, p1, delta=10.0):
    """
    High-temperature pressure branch p+(e).

    Constructed so that:
      - p'(eHT) = 0  (pressure minimum at eHT)
      - p'(e)  → 1/3 as e → ∞  (conformal limit)
      - p(eHT) = p1

    Parameters
    ----------
    e     : array_like  Energy density.
    eHT   : float       Energy at the HT turning point.
    p1    : float       Pressure value at eHT.
    delta : float       Smoothing scale (default 10).

    Returns
    -------
    ndarray or float
        Pressure p+(e).  Values for e < eHT − 0.1 are set to NaN.
    """
    e = np.asarray(e)
    p_raw = (1.0 / 3.0) * delta * np.log(np.cosh((e - eHT) / delta))
    C = p1 - (1.0 / 3.0) * delta * np.log(np.cosh(0.0))
    p = p_raw + C
    p = np.where(e < eHT*0.9, np.nan, p)
    return p if p.shape != () else p.item()


def pminus(e, eTL, n=1):
    """
    Low-temperature pressure branch p−(e).

    Constructed so that:
      - p(0) = 0
      - p'(0) = 1/3
      - p'(eTL) = 0
      - p'(e) strictly decreasing

    Parameters
    ----------
    e   : array_like  Energy density.
    eTL : float       Energy at the LT turning point.
    n   : float       Shape exponent (default 1).

    Returns
    -------
    ndarray or float
        Pressure p−(e).  Values for e > eTL + 0.1 are set to NaN.
    """
    e = np.asarray(e)
    p = np.full_like(e, np.nan, dtype=float)
    mask = e <= eTL*1.1
    p[mask] = (1.0 / 3.0) * (
        e[mask] - e[mask] ** (n + 1) / ((n + 1) * eTL ** n)
    )
    return p if p.shape != () else p.item()


# ---------------------------------------------------------------------------
# Speed of sound
# ---------------------------------------------------------------------------

def speed_of_sound_squared(e_array, p_array):
    """
    Compute c_s²(e) = dp/de from tabulated (e, p) data.

    Parameters
    ----------
    e_array : array_like  Energy density grid (must be strictly increasing).
    p_array : array_like  Pressure values on the same grid.

    Returns
    -------
    CubicSpline
        Callable cs²(e).
    """
    e_array = np.asarray(e_array)
    p_array = np.asarray(p_array)
    cs2_vals = np.gradient(p_array, e_array)
    return CubicSpline(e_array, cs2_vals)


# ---------------------------------------------------------------------------
# Custom EoS loader
# ---------------------------------------------------------------------------

def _cs2_roots(spl_cs2, n_rows, n_scan=20000):
    """All zeros of c_s² along the row index, sorted by index."""
    idxs = np.linspace(0, n_rows - 1, n_scan)
    vals = spl_cs2(idxs)
    roots = []
    for i in np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]:
        try:
            roots.append(brentq(spl_cs2, idxs[i], idxs[i + 1]))
        except ValueError:
            continue
    return np.unique(np.round(roots, 12))


def load_custom_eos(filepath, columns=(0, 1, 2, 3), idx_min=0, n_branch=100000,
                    Tc=None, cs2_from='pe', delimiter=None):
    """
    Load a tabulated equation of state and split it into its HT and LT branches.

    The file tabulates the whole equilibrium curve (stable, metastable and
    unstable parts) with columns for p and e, and optionally T and s.
    Without T and s columns they are rebuilt from p(e): de = T ds and
    e + p = T s give d ln s = de / (e + p), integrated along the whole curve,
    and T = (e + p) / s.  This fixes s and T up to one constant factor, which
    does not affect which solutions are allowed; it is chosen so that T_c = 1.
    The table is split as in the bubbles-eos notebooks:

      1. Order the rows from the HT end (largest e) and drop repeated e values.
      2. Interpolate e, p, s, T and c_s² against the row index.
      3. The turning points are zeros of c_s²: eTH is the first one coming
         from the HT end, eTL the hottest one after it (the LT spinodal),
         which skips spurious zeros in noisy low-temperature tails; the LT
         branch stops at the next zero, if any.
      4. Re-interpolate each branch as a function of e, keeping the part
         where e is monotonic away from its turning point.  The pressure
         is the antiderivative of c_s², so that p' = c_s² holds exactly.

    Parameters
    ----------
    filepath  : str    Path to the data file (whitespace or CSV).
    columns   : tuple  Column indices of (T, p, e, s) (default (0, 1, 2, 3)).
                       Use None for T and s to rebuild them from p(e), e.g.
                       (None, 1, 0, None) for a two-column (e, p) file.
    idx_min   : int    First row (after ordering from the HT end) kept on the
                       HT branch; raise it to drop unreliable rows at the
                       high-energy end (default 0).
    n_branch  : int    Resampling points per branch (default 100000).
    Tc        : float  Critical temperature, if known (e.g. 1 for tables in
                       units of T_c).  By default it is located from the
                       crossing of the free energies f = −p of both branches.
                       Only meaningful when T is read from the file.
    cs2_from  : str    'pe' for c_s² = dp/de (default), or 'Ts' for
                       c_s² = (s/T) dT/ds, which is the same quantity by
                       dp = s dT, de = T ds.  Use 'Ts' when p (and e) are
                       noisy but T and s are not, e.g. holographic tables
                       at low T; use 'pe' when p was built from a smooth
                       c_s² and T does not satisfy dp = s dT.
    delimiter : str    Column delimiter (default: whitespace).

    Returns
    -------
    dict of keyword arguments for ``interactive_bubble_plot``:
      'eTH', 'eTL'                  : float  Turning-point energies.
      'nucleation_energies_allowed' : (e_lo, e_hi)  Metastable HT range
                                      [eTH, e+(T_c)] (whole branch if no T_c).
      'bubble_energies_allowed'     : (e_lo, e_hi)  LT range [e_min, e−(T_c)].
      'custom_pplus', 'custom_pminus'
                                    : callable(e, ...)  Pressure on each branch,
                                      with the call signature of the analytic p±.
      'custom_cs2_plus', 'custom_cs2_minus'
                                    : PchipInterpolator  c_s²(e) on each branch.
      'splus', 'sminus'             : PchipInterpolator  Entropy density s(e).
      'tplus', 'tminus'             : PchipInterpolator  Temperature T(e).

    All interpolants return NaN outside their branch.

    Examples
    --------
    >>> interactive_bubble_plot(**load_custom_eos("Data_holo/thermo_phiM1.txt",
    ...                                           columns=(0, 6, 7, 1)))
    """
    data = np.loadtxt(filepath, delimiter=delimiter)
    if columns[1] is None or columns[2] is None:
        raise ValueError("Custom EoS needs p and e columns.")
    if data.ndim != 2 or data.shape[1] <= max(c for c in columns if c is not None):
        raise ValueError(
            f"Custom EoS file needs columns for (T, p, e, s) at indices {columns}."
        )
    T, p, e, s = (None if c is None else data[:, c] for c in columns)
    rebuild = T is None or s is None
    if rebuild and Tc is not None:
        raise ValueError("Tc can only be given together with T and s columns.")

    # Start the curve at its HT end (largest e) and drop repeated e values
    if e[0] < e[-1]:
        T, p, e, s = (None if a is None else a[::-1] for a in (T, p, e, s))
    keep = np.concatenate([[True], np.diff(e) != 0])
    if not rebuild:
        keep &= np.concatenate([[True], np.diff(s) != 0])
    T, p, e, s = (None if a is None else a[keep] for a in (T, p, e, s))

    if rebuild:
        # d ln s = de / (e + p) along the whole curve; the constant factor is
        # fixed below so that T_c = 1
        s = np.exp(cumulative_trapezoid(1.0 / (e + p), e, initial=0.0))
        T = (e + p) / s

    if cs2_from == 'pe':
        cs2 = np.gradient(p, e)
    elif cs2_from == 'Ts':
        cs2 = (s / T) * np.gradient(T, s)
    else:
        raise ValueError("cs2_from must be 'pe' or 'Ts'.")

    # Interpolate against the row index, which is monotone along the curve
    idx = np.arange(len(e), dtype=float)
    spl_e   = PchipInterpolator(idx, e)
    spl_p   = PchipInterpolator(idx, p)
    spl_s   = PchipInterpolator(idx, s)
    spl_T   = PchipInterpolator(idx, T)
    spl_cs2 = PchipInterpolator(idx, cs2)

    roots = _cs2_roots(spl_cs2, len(e))
    if len(roots) < 2:
        raise ValueError("Custom EoS: could not find two c_s² = 0 turning points.")
    eTH_idx = roots[0]
    eTL_idx = roots[1:][np.argmax(spl_T(roots[1:]))]

    def _branch(i_turn, i_far):
        ii = np.linspace(i_turn, i_far, n_branch)
        e_b, p_b, s_b, T_b, cs2_b = (f(ii) for f in (spl_e, spl_p, spl_s, spl_T, spl_cs2))
        # Keep the part where e is monotonic away from the turning point
        de = np.diff(e_b)
        bad = np.where(de * de[0] <= 0)[0]
        if len(bad) > 0:
            e_b, p_b, s_b, T_b, cs2_b = (a[:bad[0] + 1] for a in (e_b, p_b, s_b, T_b, cs2_b))
        e_turn, p_turn = e_b[0], p_b[0]
        if e_b[0] > e_b[-1]:
            e_b, p_b, s_b, T_b, cs2_b = (a[::-1] for a in (e_b, p_b, s_b, T_b, cs2_b))
        cs2_of_e = PchipInterpolator(e_b, cs2_b, extrapolate=False)
        anti = cs2_of_e.antiderivative()
        # Fix the constant at the turning point: integration errors grow away
        # from it, toward the sparse ends of the table that matter least
        offset = float(p_turn - anti(e_turn))
        p_of_e = lambda x, *args, **kwargs: anti(x) + offset
        return (p_of_e, cs2_of_e,
                PchipInterpolator(e_b, s_b, extrapolate=False),
                PchipInterpolator(e_b, T_b, extrapolate=False),
                (float(e_b[0]), float(e_b[-1])))

    # The LT branch ends at the next zero of c_s², if any (noisy cold tails)
    later = roots[roots > eTL_idx]
    pplus_t, cs2_plus, splus, tplus, e_plus = _branch(eTH_idx, idx_min)
    pminus_t, cs2_minus, sminus, tminus, e_minus = _branch(
        eTL_idx, later[0] if len(later) > 0 else len(e) - 1)
    eTH, eTL = e_plus[0], e_minus[1]

    # Critical temperature: crossing of the free energies f = −p
    if Tc is None:
        from .thermodynamics import find_critical_temperature  # avoid circular import
        e_m, e_p = tminus.x, tplus.x   # branch knots: as dense as the data
        Tc = find_critical_temperature(tminus(e_m), -pminus_t(e_m), tplus(e_p), -pplus_t(e_p))
    if rebuild and Tc is not None:
        # Fix the free constant of the rebuilt T and s: T in units of T_c
        tplus, tminus = (PchipInterpolator(f.x, f(f.x) / Tc, extrapolate=False)
                         for f in (tplus, tminus))
        splus, sminus = (PchipInterpolator(f.x, f(f.x) * Tc, extrapolate=False)
                         for f in (splus, sminus))
        Tc = 1.0
    eN_range, eC_range = e_plus, e_minus
    if Tc is not None:
        try:
            eN_range = (eTH, brentq(lambda x: tplus(x) - Tc, *e_plus))
            eC_range = (e_minus[0], brentq(lambda x: tminus(x) - Tc, *e_minus))
        except ValueError:
            pass

    print(
        f"[snobex] Custom EoS loaded from '{filepath}': "
        f"HT branch e in [{e_plus[0]:.4g}, {e_plus[1]:.4g}], "
        f"LT branch e in [{e_minus[0]:.4g}, {e_minus[1]:.4g}], "
        + (f"T_c = {Tc:.4g}" if Tc is not None else "no T_c found")
        + (" (T, s rebuilt from p(e))" if rebuild else "")
        + f"; E_N in [{eN_range[0]:.4g}, {eN_range[1]:.4g}], "
        f"E_C in [{eC_range[0]:.4g}, {eC_range[1]:.4g}]"
    )
    return {
        'eTH': eTH, 'eTL': eTL,
        'nucleation_energies_allowed': eN_range,
        'bubble_energies_allowed': eC_range,
        'custom_pplus': pplus_t, 'custom_pminus': pminus_t,
        'custom_cs2_plus': cs2_plus, 'custom_cs2_minus': cs2_minus,
        'splus': splus, 'sminus': sminus,
        'tplus': tplus, 'tminus': tminus,
    }
