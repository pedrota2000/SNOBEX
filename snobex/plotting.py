"""
snobex.plotting
================
Interactive phase-space bubble plots for snobex.

``interactive_bubble_plot`` (supercooled transitions) computes all
separator curves and hydrodynamic flows, then displays a live
matplotlib figure where clicking on any point shows the corresponding
velocity and energy profiles.  ``superheated_bubble_plot`` does the same
for superheated transitions, so far with inverse deflagrations and
inverse detonations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import concurrent.futures
import functools
import pickle
import os

from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

from .eos import Branch, pplus as _pplus_analytic, pminus as _pminus_analytic
from .hydrodynamics import system_minus, pressure_balance_energy
from .supercooled import (
    find_shocked_detonation, find_shocked_detonation_jouguet,
    shocked_detonation_strip, find_deto, find_def, find_hyb,
    _e_max_from_pressure
)
from .supercooled_separators import (
    compute_def_separator,
    compute_det_separator,
    compute_hyb_separator,
    limiting_detonation_contour_finder,
    jouguet_detonation_contour_finder,
    compute_entropy_separator_det,
    compute_entropy_separator_def,
    compute_alphan,
    eN_contour_finder,
    eC_contour_finder,
    entropy_checker,
)
from .superheated import (
    scan_inverse_deflagrations,
    scan_inverse_detonations,
    superheated_window,
    inverse_def_entropy_jump,
    inverse_det_entropy_jump,
    alpha_n,
)
from .superheated_separators import (
    inverse_jouguet_contour_finder,
    limiting_inverse_def_contour_finder,
    spinodal_inverse_det_contour_finder,
    sonic_inverse_det_contour_finder,
    dcp_energy,
)


def interactive_bubble_plot(eTH, eTL, nucleation_energies_allowed, bubble_energies_allowed,
                          p0=0.0, p1=0.0, xiw_resolution=25, en_resolution=50,
                          contour_resolution=500, show_eos=True,
                          sminus=None, splus=None, tminus=None, tplus=None, n=4.0, delta=1.0,
                          sTL=None, sTH=None, saving=False, load_eos=None,
                          custom_pplus=None, custom_pminus=None,
                          custom_cs2_plus=None, custom_cs2_minus=None,
                          output_dir="."):
    """
    Compute and display the interactive phase-space bubble plot.

    Parameters
    ----------
    eTH, eTL : float
        HT/LT turning-point energies.
    nucleation_energies_allowed : tuple (eN_lo, eN_hi)
        Allowed nucleation energy range.
    bubble_energies_allowed : tuple (eC_lo, eC_hi)
        Allowed bubble energy range.
    p0, p1 : float, optional
        EoS pressure offsets (default 0.0).
    xiw_resolution : int
        Number of wall-velocity grid points (default 25).
    en_resolution : int
        Number of energy grid points per axis (default 50).
    show_eos : bool
        Show EoS plot before computing (default True).
    sminus, splus : callable, optional
        Entropy interpolants S−(E), S+(E) for entropy-condition checks.
    n : float
        LT branch shape exponent.
    delta : float
        HT branch smoothing scale.
    sTL, sTH : float, optional
        Entropy anchor values (used only when ``saving=True``).
    saving : bool
        Pickle the results to ``output_dir`` (default False).
    load_eos : str, optional
        Path to a previously saved ``.pkl`` file; overrides all EoS parameters.
    custom_pplus : callable(e, eHT, p1), optional
        Replacement for the built-in HT pressure function.
    custom_pminus : callable(e, eTL, p0), optional
        Replacement for the built-in LT pressure function.
    custom_cs2_plus : callable(e), optional
        Replacement for the built-in HT speed-of-sound-squared.
    custom_cs2_minus : callable(e), optional
        Replacement for the built-in LT speed-of-sound-squared.
    output_dir : str
        Directory for output files (default current directory).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes  (the (ξ_w, E_N) axis)
    """

    # ------------------------------------------------------------------
    # Optionally load EoS parameters from a saved pickle
    # ------------------------------------------------------------------
    results_dict = {
        'EoS_parameters': {
            'eTL': eTL, 'eTH': eTH, 'p0': p0, 'p1': p1,
            'n': n, 'delta': delta, 'sTL': sTL, 'sTH': sTH,
        },
        'Global_contours': {},
        'Local_contours': {},
        'Hydro_flows': {},
    }

    if load_eos is not None:
        print(f"[snobex] Loading EoS from '{load_eos}'")
        with open(load_eos, "rb") as f:
            results_dict = pickle.load(f)
        ep = results_dict['EoS_parameters']
        eTL = ep['eTL'];  eTH   = ep['eTH']
        p0  = ep['p0'];   p1    = ep['p1']
        n   = ep['n'];    delta = ep['delta']
        sTL = ep['sTL'];  sTH   = ep['sTH']

    # ------------------------------------------------------------------
    # Resolve EoS callables (analytic or custom)
    # ------------------------------------------------------------------
    if custom_pplus is not None:
        pplus  = custom_pplus
    else:
        def pplus(e, eHT, p1_arg=p1, delta=delta):
            return _pplus_analytic(e, eHT, p1_arg, delta)

    if custom_pminus is not None:
        pminus = custom_pminus
    else:
        def pminus(e, eTL_arg, p0_arg=p0, n=n):
            return _pminus_analytic(e, eTL_arg, n)

    # ------------------------------------------------------------------
    # Energy scale
    # ------------------------------------------------------------------
    if custom_pplus is not None:
        # Tabulated EoS (NaN outside its branches): bracket the root instead
        e_max = _e_max_from_pressure(eTH, eTL, p0, p1, pminus, pplus)
    elif eTL > eTH:
        e_max = fsolve(lambda e: pminus(eTL, eTL, p0) - pplus(e, eTH, p1), eTL * 2)[0]
    else:
        e_max = fsolve(lambda e: pminus(eTL, eTL, p0) - pplus(e, eTH, p1), eTH * 5)[0]
    e_max *= 2.0
    if custom_pplus is not None:
        # ...and do not go past the top of the tabulated HT branch
        e_scan = np.linspace(eTH, e_max, 2000)
        e_max = float(e_scan[np.isfinite(pplus(e_scan, eTH, p1))].max())
    print(f"[snobex] e_max = {e_max:.4f}")

    # ------------------------------------------------------------------
    # Speed of sound
    # ------------------------------------------------------------------
    if custom_cs2_plus is not None:
        cs2_plus = custom_cs2_plus
    else:
        e_plus  = np.linspace(eTH * 0.9, e_max * 5, 5000)
        p_plus  = pplus(e_plus, eTH, p1)
        cs2_plus = CubicSpline(e_plus, np.gradient(p_plus, e_plus))

    if custom_cs2_minus is not None:
        cs2_minus = custom_cs2_minus
    else:
        e_minus  = np.linspace(0.0, eTL * 1.1, 5000)
        p_minus  = pminus(e_minus, eTL, p0)
        cs2_minus = CubicSpline(e_minus, np.gradient(p_minus, e_minus))

    # ------------------------------------------------------------------
    # The two phases.  Supercooled transition: the LT bubble grows into the
    # metastable HT phase.  The solvers below take the individual callables,
    # which are passed on from the branches.
    # ------------------------------------------------------------------
    outside = Branch('HT', p=lambda e, f=pplus: f(e, eTH, p1), cs2=cs2_plus,
                     e_turn=eTH, up=True, s=splus, T=tplus)
    inside = Branch('LT', p=lambda e, f=pminus: f(e, eTL, p0), cs2=cs2_minus,
                    e_turn=eTL, up=False, s=sminus, T=tminus)
    pplus, pminus = outside.pressure, inside.pressure
    cs2_plus, cs2_minus = outside.cs2, inside.cs2

    # ------------------------------------------------------------------
    # Inflection-point equation (needed for hybrid separators)
    # ------------------------------------------------------------------
    def inf_point_eq(e):
        first  = 2.0 * (cs2_minus(e) - 1.0) * cs2_minus(e)
        second = -(e + pminus(e, eTL, p0)) * cs2_minus(e, nu=1)
        return first + second

    # ------------------------------------------------------------------
    # Optionally show EoS
    # ------------------------------------------------------------------
    if show_eos:
        _plot_eos(pplus, pminus, eTH, eTL, p0, p1, e_max)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    xiw_vals = np.linspace(0.01, 0.99, xiw_resolution)
    nucleation_energies = np.linspace(eTH + 1e-3, e_max, en_resolution)
    bubble_energies     = np.linspace(1e-6, eTL - 1e-3, en_resolution)

    # ------------------------------------------------------------------
    # Figure scaffold
    # ------------------------------------------------------------------
    fig_det = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.3, wspace=0.2)
    ax_det   = fig_det.add_subplot(gs[0, 0])
    ax_C     = fig_det.add_subplot(gs[0, 1])
    ax_alpha = fig_det.add_subplot(gs[1, :])

    for ax, xlabel, ylabel in [
        (ax_det,   r'$\xi_w$', r'$\mathcal{E}_N$'),
        (ax_C,     r'$\xi_w$', r'$\mathcal{E}_C$'),
        (ax_alpha, r'$\xi_w$', r'$\alpha_N$'),
    ]:
        ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel);  ax.grid(True)
    ax_det.set_xlim(-0.01, 1.01)
    fig_det.suptitle("Computing separators…")
    plt.show(block=False)
    fig_det.canvas.draw(); fig_det.canvas.flush_events()

    # Helper: alpha_N with current EoS parameters
    def _alphan(eN):
        if sminus is None or splus is None:
            return np.nan
        return compute_alphan(eN, pplus, pminus, eTH, eTL, p0, p1,
                               splus, sminus, n, delta, tplus=tplus, tminus=tminus)

    # ------------------------------------------------------------------
    # Separator curves (parallel)
    # ------------------------------------------------------------------
    xi_temporal = np.linspace(1e-2, 0.99, contour_resolution * 10)
    eN_temporal = np.linspace(eTH, e_max, contour_resolution * 10)

    # Deflagration separator
    fig_det.suptitle("Computing deflagration separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: compute_def_separator(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            xi_temporal))
    def_eC = np.array([r[0] for r in results], dtype=float)
    def_eN = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(def_eC) & ~np.isnan(def_eN)
    def_eC, def_eN, xi_def = def_eC[mask], def_eN[mask], xi_temporal[mask]
    ax_C.plot(xi_def, def_eC, '--k')
    ax_det.plot(xi_def, def_eN, '--k')
    ax_alpha.plot(xi_def, np.array(list(map(_alphan, def_eN))), '--k')
    results_dict['Global_contours']['deflagration'] = {
        'xi': xi_def, 'eC': def_eC, 'eN': def_eN,
        'alpha': np.array(list(map(_alphan, def_eN)))}

    # Detonation separator
    fig_det.suptitle("Computing detonation separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda eN: compute_det_separator(
                eN, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            eN_temporal))
    det_xiw = np.array([r[0] for r in results], dtype=float)
    det_eC  = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(det_xiw) & ~np.isnan(det_eC)
    det_xiw, det_eC, det_eN = det_xiw[mask], det_eC[mask], np.array(eN_temporal)[mask]
    ax_det.plot(det_xiw, det_eN, '--k')
    ax_C.plot(det_xiw, det_eC, '--k')
    ax_alpha.plot(det_xiw, np.array(list(map(_alphan, det_eN))), '--k')
    results_dict['Global_contours']['detonation'] = {
        'xi': det_xiw, 'eC': det_eC, 'eN': det_eN,
        'alpha': np.array(list(map(_alphan, det_eN)))}

    # Hybrid separator
    fig_det.suptitle("Computing hybrid separators…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: compute_hyb_separator(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus, inf_point_eq),
            xi_temporal))
    hyb_eN = np.array([r[0] for r in results], dtype=float)
    hyb_eC = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(hyb_eN) & ~np.isnan(hyb_eC)
    hyb_eN, hyb_eC, xi_hyb = hyb_eN[mask], hyb_eC[mask], xi_temporal[mask]
    ax_C.plot(xi_hyb, hyb_eC, '--g')
    ax_det.plot(xi_hyb, hyb_eN, '--g')
    ax_alpha.plot(xi_hyb, np.array(list(map(_alphan, hyb_eN))), '--g')
    results_dict['Local_contours']['hybrid'] = {
        'xi': xi_hyb, 'eC': hyb_eC, 'eN': hyb_eN,
        'alpha': np.array(list(map(_alphan, hyb_eN)))}

    # Limiting detonation contour
    fig_det.suptitle("Computing limiting detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    xiw_v2 = np.linspace(0.3, 1 - 1e-2, int(contour_resolution))
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda xiw: limiting_detonation_contour_finder(
                xiw, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus, inf_point_eq),
            xiw_v2))
    ld_xish = np.array([r[0] for r in results], dtype=float)
    ld_eN   = np.array([r[1] for r in results], dtype=float)
    ld_eC   = np.array([r[2] for r in results], dtype=float)
    m1 = ~(np.isnan(ld_eN) | np.isnan(ld_xish))
    m2 = ~(np.isnan(ld_eC) | np.isnan(ld_xish))
    ax_det.plot(ld_xish[m1], ld_eN[m1], '--r', lw=2)
    ax_C.plot(ld_xish[m2],   ld_eC[m2], '--r', lw=2)
    ax_alpha.plot(ld_xish[m1], np.array(list(map(_alphan, ld_eN[m1]))), '--r', lw=2)
    results_dict['Local_contours']['detonation'] = {
        'xi': ld_xish[m1], 'eC': ld_eC[m2], 'eN': ld_eN[m1],
        'alpha': np.array(list(map(_alphan, ld_eN[m1])))}

    # Jouguet detonations
    fig_det.suptitle("Computing Jouguet detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    eN_jouguet = np.linspace(eTH * 1.01, e_max, contour_resolution)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(
            lambda eN: jouguet_detonation_contour_finder(
                eN, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            eN_jouguet))
    j_xiw = np.array([r[0] for r in results], dtype=float)
    j_eC  = np.array([r[1] for r in results], dtype=float)
    mask = ~np.isnan(j_xiw) & ~np.isnan(j_eC)
    j_xiw, j_eC, j_eN = j_xiw[mask], j_eC[mask], eN_jouguet[mask]
    max_hyb_xiw = np.max(xi_hyb) if len(xi_hyb) > 0 else 0.0
    final_mask = j_xiw >= max_hyb_xiw
    j_xiw, j_eC, j_eN = j_xiw[final_mask], j_eC[final_mask], j_eN[final_mask]
    ax_det.plot(j_xiw, j_eN, '--b', lw=2)
    ax_C.plot(j_xiw, j_eC, '--b', lw=2)
    ax_alpha.plot(j_xiw, np.array(list(map(_alphan, j_eN))), '--b', lw=2)
    results_dict['Local_contours']['jouguet'] = {
        'xi': j_xiw, 'eC': j_eC, 'eN': j_eN,
        'alpha': np.array(list(map(_alphan, j_eN)))}

    # # Entropy separator — detonations
    # fig_det.suptitle("Computing entropy detonation contour…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    # if sminus is not None and splus is not None:
    #     xiw_entr = np.linspace(0.3, 1 - 1e-3, contour_resolution)
    #     with concurrent.futures.ThreadPoolExecutor() as ex:
    #         results = list(ex.map(
    #             lambda xiw: compute_entropy_separator_det(
    #                 xiw, pplus, pminus, eTH, eTL, p0, p1,
    #                 cs2_plus, cs2_minus, splus, sminus, e_max),
    #             xiw_entr))
    #     entr_det_eN = np.array([r[0] for r in results], dtype=float)
    #     entr_det_eC = np.array([r[1] for r in results], dtype=float)
    #     mask = ~np.isnan(entr_det_eN) & ~np.isnan(entr_det_eC)
    #     ax_det.plot(xiw_entr[mask], entr_det_eN[mask], ':b', lw=2)
    #     ax_C.plot(xiw_entr[mask],   entr_det_eC[mask], ':b', lw=2)
    #     ax_alpha.plot(xiw_entr[mask], np.array(list(map(_alphan, entr_det_eN[mask]))), ':b', lw=2)
    #
    #     # Entropy separator — deflagrations
    #     fig_det.suptitle("Computing entropy deflagration contour…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    #     xiw_entr = np.linspace(0.01, 1 - 1e-3, contour_resolution)
    #     with concurrent.futures.ThreadPoolExecutor() as ex:
    #         results = list(ex.map(
    #             lambda xiw: compute_entropy_separator_def(
    #                 xiw, pplus, pminus, eTH, eTL, p0, p1,
    #                 cs2_plus, cs2_minus, splus, sminus, e_max),
    #             xiw_entr))
    #     entr_def_eC = np.array([r[0] for r in results], dtype=float)
    #     entr_def_eN = np.array([r[1] for r in results], dtype=float)
    #     mask = ~np.isnan(entr_def_eN) & ~np.isnan(entr_def_eC)
    #     ax_C.plot(xiw_entr[mask],   entr_def_eC[mask], ':b', lw=2)
    #     ax_det.plot(xiw_entr[mask], entr_def_eN[mask], ':b', lw=2)
    #     ax_alpha.plot(xiw_entr[mask], np.array(list(map(_alphan, entr_def_eN[mask]))), ':b', lw=2)

    fig_det.suptitle(None); fig_det.canvas.draw()

    # ------------------------------------------------------------------
    # Progress bars
    # ------------------------------------------------------------------
    bh, bs, by = 0.015, 0.003, 0.94
    ax_bar_det = fig_det.add_axes([0.15, by, 0.7, bh])
    ax_bar_def = fig_det.add_axes([0.15, by - (bh + bs), 0.7, bh])
    ax_bar_hyb = fig_det.add_axes([0.15, by - 2 * (bh + bs), 0.7, bh])
    ax_bar_sds = fig_det.add_axes([0.15, by - 3 * (bh + bs), 0.7, bh])
    for axb in (ax_bar_det, ax_bar_def, ax_bar_hyb, ax_bar_sds):
        axb.set_xticks([]);  axb.set_yticks([])
        axb.set_xlim(0, 1);  axb.set_ylim(0, 1)
    det_rect = ax_bar_det.barh([0.5], [0], height=1.0, color='red')[0]
    def_rect = ax_bar_def.barh([0.5], [0], height=1.0, color='blue')[0]
    hyb_rect = ax_bar_hyb.barh([0.5], [0], height=1.0, color='green')[0]
    sds_rect = ax_bar_sds.barh([0.5], [0], height=1.0, color='purple')[0]
    for axb, rect, label in [
        (ax_bar_det, det_rect, "Detonations"),
        (ax_bar_def, def_rect, "Deflagrations"),
        (ax_bar_hyb, hyb_rect, "Hybrids"),
        (ax_bar_sds, sds_rect, "Shocked det."),
    ]:
        axb.text(-0.02, 0.5, label, va='center', ha='right', fontsize=8,
                 transform=axb.transAxes)
        axb.text(1.02, 0.5, "0%", va='center', ha='left', fontsize=8,
                 transform=axb.transAxes, gid=label + '_pct')

    def _pct_text(ax, label):
        for t in ax.texts:
            if t.get_gid() == label + '_pct':
                return t
        return None

    det_pct = _pct_text(ax_bar_det, "Detonations")
    def_pct = _pct_text(ax_bar_def, "Deflagrations")
    hyb_pct = _pct_text(ax_bar_hyb, "Hybrids")
    sds_pct = _pct_text(ax_bar_sds, "Shocked det.")

    def _update_bar(rect, pct_text, frac, ax, total, done):
        rect.set_width(frac)
        if pct_text:
            pct_text.set_text(f"{frac*100:.1f}%")
        if done % max(1, total // 25) == 0:
            ax.figure.canvas.draw_idle()
            fig_det.canvas.flush_events()

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------
    def _det_worker(args):
        xiw, e = args
        __, flow = find_deto(e, xiw, pplus, pminus, eTH, eTL, p0, p1,
                              cs2_plus, cs2_minus)
        return xiw, e, flow

    def _def_worker(args):
        xiw, eC = args
        eN, flow = find_def(eC, xiw, pplus, pminus, eTH, eTL, p0, p1,
                             cs2_plus, cs2_minus)
        return xiw, eC, eN, flow

    def _hyb_worker(args):
        xiw, em = args
        eN, flow, emwall = find_hyb(em, xiw, pplus, pminus, eTH, eTL, p0, p1,
                                     cs2_plus, cs2_minus)
        return xiw, eN, flow, emwall
    
    def _shocked_det_worker(args):
        xiw, eN = args
        # Definitive finder: selects the second shock via the Jouguet condition.
        # The legacy find_shocked_detonation (family of shocks) is kept in
        # snobex.hydrodynamics for reference.
        result = find_shocked_detonation_jouguet(eN, xiw, pplus, pminus,
                                                 eTH, eTL, p0, p1,
                                                 cs2_plus, cs2_minus)
        return xiw, eN, result

    # ------------------------------------------------------------------
    # Compute flows
    # ------------------------------------------------------------------
    detonation_flows        = []
    deflagration_flows      = []
    hybrid_flows            = []
    shocked_detonation_flows = []

    # Detonations
    XIW, E = np.meshgrid(xiw_vals, nucleation_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, e, flow in ex.map(_det_worker, tasks):
            done += 1
            _update_bar(det_rect, det_pct, done / total, ax_bar_det, total, done)
            if flow is not None and not np.isnan(flow).any():
                v_f, xi_f, e_f = flow
                detonation_flows.append((xiw, e, float(min(e_f)), v_f, xi_f, e_f))
    _batch_plot(ax_det, ax_C, ax_alpha, detonation_flows, 'ro', 'Detonations',
                5, _alphan)
    results_dict['Hydro_flows']['detonation'] = _pack_flows(detonation_flows, _alphan)

    # Deflagrations
    XIW, E = np.meshgrid(xiw_vals, bubble_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, eC, eN, flow in ex.map(_def_worker, tasks):
            done += 1
            _update_bar(def_rect, def_pct, done / total, ax_bar_def, total, done)
            if flow is None or np.isnan(eN):
                continue
            v_f, xi_f, e_f = flow
            deflagration_flows.append((xiw, eN, eC, v_f, xi_f, e_f))
    _batch_plot(ax_det, ax_C, ax_alpha, deflagration_flows, 'bo', 'Deflagration',
                4, _alphan)
    results_dict['Hydro_flows']['deflagration'] = _pack_flows(deflagration_flows, _alphan)

    # Hybrids
    XIW, E = np.meshgrid(xiw_vals, bubble_energies, indexing='ij')
    tasks = [(XIW[i, j], E[i, j]) for i in range(XIW.shape[0]) for j in range(XIW.shape[1])]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, eN, flow, emwall in ex.map(_hyb_worker, tasks):
            done += 1
            _update_bar(hyb_rect, hyb_pct, done / total, ax_bar_hyb, total, done)
            if flow is None or np.isnan(eN):
                continue
            v_f, xi_f, e_f = flow
            hybrid_flows.append((xiw, eN, float(min(e_f)), v_f, xi_f, e_f, emwall))
    _batch_plot(ax_det, ax_C, ax_alpha, hybrid_flows, 'go', 'Hybrids', 3, _alphan)
    results_dict['Hydro_flows']['hybrid'] = _pack_flows(hybrid_flows, _alphan)

    # Shocked detonations live in a thin strip of wall speeds just below the
    # ordinary detonations, usually narrower than a grid cell: locate the
    # strip for each E_N and sample ξ_w inside it.  Each worker returns
    # parallel lists.  Plotting/packing is deferred until after the entropy
    # check, which decides which candidates survive.
    en_sds = np.linspace(nucleation_energies[0], nucleation_energies[-1], en_resolution)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        strips = list(ex.map(
            lambda eN: shocked_detonation_strip(
                eN, pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus),
            en_sds))
    tasks = [(xiw, eN) for eN, (lo, hi) in zip(en_sds, strips) if hi > lo
             for xiw in np.linspace(lo, hi, xiw_resolution)]
    total = len(tasks);  done = 0
    with concurrent.futures.ThreadPoolExecutor() as ex:
        for xiw, eN, result in ex.map(_shocked_det_worker, tasks):
            done += 1
            _update_bar(sds_rect, sds_pct, done / total, ax_bar_sds, total, done)
            if not isinstance(result, tuple) or len(result) != 5:
                continue
            eC_list_r, eN_list_r, xiw_list_r, flow_list_r, xish_list_r = result
            for eC_r, eN_r, xiw_r, flow_r, xish_r in zip(
                    eC_list_r, eN_list_r, xiw_list_r, flow_list_r, xish_list_r):
                if xiw_r is None or np.isnan(eC_r) or np.isnan(eN_r) or np.isnan(xiw_r):
                    continue
                v_f, xi_f, e_f = flow_r
                if np.isnan(v_f).any():
                    continue
                shocked_detonation_flows.append(
                    (xiw_r, eN_r, eC_r, v_f, xi_f, e_f, xish_r))

    # Remove progress bars
    ax_bar_det.remove();  ax_bar_def.remove();  ax_bar_hyb.remove(); ax_bar_sds.remove()

    # ------------------------------------------------------------------
    # Allowed-energy contours
    # ------------------------------------------------------------------
    eN_plus, eN_minus = eN_contour_finder(
        xiw_vals, bubble_energies_allowed,
        detonation_flows, deflagration_flows, hybrid_flows,
        pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)
    eC_plus, eC_minus = eC_contour_finder(
        xiw_vals, nucleation_energies_allowed,
        detonation_flows, deflagration_flows, hybrid_flows,
        pplus, pminus, eTH, eTL, p0, p1, cs2_plus, cs2_minus)

    ax_det.axhspan(*nucleation_energies_allowed, color='orange', alpha=0.2,
                   label=r'Allowed $\mathcal{E}_N$')
    if len(eN_plus) > 0 and len(eN_minus) > 0:
        ax_det.fill_between(eN_plus[:, 0], eN_plus[:, 1], eN_minus[:, 1],
                            color='green', alpha=0.2, label=r'Allowed $\mathcal{E}_C$')
    ax_C.axhspan(*bubble_energies_allowed, color='green', alpha=0.2,
                 label=r'Allowed $\mathcal{E}_C$')
    if len(eC_plus) > 0 and len(eC_minus) > 0:
        ax_C.fill_between(eC_plus[:, 0], eC_plus[:, 1], eC_minus[:, 1],
                          color='orange', alpha=0.2, label=r'Allowed $\mathcal{E}_N$')

    # ------------------------------------------------------------------
    # Entropy filtering
    # ------------------------------------------------------------------
    if sminus is not None and splus is not None:
        (bad_det, bad_def, bad_def_sh,
         bad_hyb, bad_hyb_sh,
         bad_dsd, bad_dsd_sh) = entropy_checker(
            detonation_flows, deflagration_flows, hybrid_flows,
            shocked_detonation_flows, splus, sminus)

        # Keep only shocked detonations whose shock satisfies the entropy
        # condition (ΔJ_s ≥ −1e-4, i.e. non-decreasing up to tolerance).
        if len(bad_dsd_sh) == len(shocked_detonation_flows) and len(bad_dsd_sh) > 0:
            keep = np.asarray(bad_dsd_sh) >= -1.0 / 1e4
            shocked_detonation_flows = [
                r for r, k in zip(shocked_detonation_flows, keep) if k]

        # _entropy_overlay(ax_det, ax_C, ax_alpha, detonation_flows,
        #                  bad_det,    'k*', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, deflagration_flows,
        #                  bad_def_sh, 'yd', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, deflagration_flows,
        #                  bad_def,    'k*', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, hybrid_flows,
        #                  bad_hyb_sh, 'yd', 2.5, _alphan)
        # _entropy_overlay(ax_det, ax_C, ax_alpha, hybrid_flows,
        #                  bad_hyb,    'k*', 2.5, _alphan)

    # Plot and record the (entropy-filtered) shocked detonations
    _batch_plot(ax_det, ax_C, ax_alpha, shocked_detonation_flows,
                'm^', 'Shocked det.', 4, _alphan)
    results_dict['Hydro_flows']['shocked_detonation'] = {
        **_pack_flows(shocked_detonation_flows, _alphan),
        'xish': np.array([r[6] for r in shocked_detonation_flows]),
    }

    # ------------------------------------------------------------------
    # Final plot decoration
    # ------------------------------------------------------------------
    for ax in (ax_det, ax_C, ax_alpha):
        ax.legend()
    ax_det.set_title(
        rf"Interactive Solutions for EoS $e_{{TL}}-e_{{TH}} = {eTL - eTH:.2f}$"
        + "\n(Click on points to see flow details)"
    )
    ax_det.set_xlim(-0.01, 1.01)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if saving:
        save_path = os.path.join(
            output_dir,
            f"snobex_results_{eTH:.3f}_{eTL:.3f}_{p0:.3f}.pkl",
        )
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"[snobex] Results saved to '{save_path}'")

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------
    fig_det.canvas.mpl_connect(
        'button_press_event',
        lambda ev: _on_click(ev, ax_det, ax_C, ax_alpha,
                             detonation_flows, deflagration_flows, hybrid_flows,
                             shocked_detonation_flows,
                             output_dir, _alphan),
    )

    plt.show()
    return fig_det, ax_det


def superheated_bubble_plot(outside, inside, xiw_resolution=25, en_resolution=25,
                            contour_resolution=25, xiw_range=None, ec_range=None,
                            saving=False, output_dir="."):
    """
    Interactive phase-space plot of a superheated transition: a bubble of the
    ``inside`` (HT) phase expanding into the metastable ``outside`` (LT)
    phase.  Same layout and conventions as ``interactive_bubble_plot``; so far
    it contains the inverse deflagrations (blue) and inverse detonations
    (red).  Those that violate the entropy condition at the wall are marked
    with black stars.  The separators have the styles of their supercooled
    counterparts: dashed blue, the inverse Jouguet deflagrations (as the
    Jouguet detonations); dashed red, the limiting inverse deflagrations (as
    the limiting detonations); dashed black, the slowest inverse detonations,
    below which the junction conditions have no solution (as the detonation
    separator), and the inverse detonations with xi_w = c_s,out(E_N), the
    border with the inverse hybrids (as the deflagration separator).

    Parameters
    ----------
    outside, inside : Branch  LT and HT branches; their s and T are needed
                              for the entropy condition, alpha_N and T_c.
    xiw_resolution, en_resolution : int  Grid laid over the solution region.
    contour_resolution : int  Each separator is sampled uniformly at
                              10 x contour_resolution points, as in
                              ``interactive_bubble_plot`` (default 25).
    xiw_range, ec_range : (lo, hi), optional
        Ranges searched for solutions.  By default xi_w runs up to the
        largest sound speed of the inside phase, and E_C from the inside
        turning point to the energy whose pressure equals that of the
        outside phase at its turning point: beyond it no outside state can
        balance the bubble (capped at the top of the inside branch).
    saving : bool  Pickle the results to ``output_dir`` (default False).
    output_dir : str  Directory for output files (default current directory).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes  (the (ξ_w, E_N) axis)
    """
    Tc, e_out_Tc, e_in_Tc = superheated_window(outside, inside)
    if ec_range is None:
        ec_top = pressure_balance_energy(inside, outside)      # p_in(E_C) = p_out(e_turn,out)
        if not np.isfinite(ec_top):
            ec_top = 6.0 * inside.e_turn
        ec_range = (inside.e_turn * 1.001, min(inside.e_hi, ec_top))
    if xiw_range is None:
        cs2 = np.asarray(inside.cs2(np.linspace(*ec_range, 200)), dtype=float)
        cs_max = float(np.sqrt(np.nanmax(cs2))) if np.isfinite(cs2).any() else 0.99
        xiw_range = (0.003, min(0.99, cs_max))
    print(f"[snobex] Superheated: T_c = {Tc:.4g}, allowed E_N in [{e_out_Tc:.4g}, {outside.e_turn:.4g}]"
          if e_out_Tc is not None else "[snobex] Superheated: no T_c found")

    results_dict = {
        'EoS_parameters': {'eTL': outside.e_turn, 'eTH': inside.e_turn, 'Tc': Tc,
                           'transition': 'superheated'},
        'Global_contours': {},
        'Local_contours': {},
        'Hydro_flows': {},
    }
    _alphan = functools.lru_cache(maxsize=None)(lambda eN: alpha_n(float(eN), outside, inside))

    # ------------------------------------------------------------------
    # Figure scaffold (as in interactive_bubble_plot)
    # ------------------------------------------------------------------
    fig_det = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.3, wspace=0.2)
    ax_det   = fig_det.add_subplot(gs[0, 0])
    ax_C     = fig_det.add_subplot(gs[0, 1])
    ax_alpha = fig_det.add_subplot(gs[1, :])

    for ax, xlabel, ylabel in [
        (ax_det,   r'$\xi_w$', r'$\mathcal{E}_N$'),
        (ax_C,     r'$\xi_w$', r'$\mathcal{E}_C$'),
        (ax_alpha, r'$\xi_w$', r'$\alpha_N$'),
    ]:
        ax.set_xlabel(xlabel);  ax.set_ylabel(ylabel);  ax.grid(True)
    ax_det.set_xlim(-0.01, 1.01)
    fig_det.suptitle("Computing separators…")
    plt.show(block=False)
    fig_det.canvas.draw(); fig_det.canvas.flush_events()

    # ------------------------------------------------------------------
    # Separator curves
    # ------------------------------------------------------------------
    # Inverse Jouguet deflagrations: the fastest walls with the centre at rest
    fig_det.suptitle("Computing inverse Jouguet deflagrations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    eC_jouguet = np.linspace(*ec_range, contour_resolution * 10)
    jouguet = lambda eC: inverse_jouguet_contour_finder(eC, outside, inside)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(jouguet, eC_jouguet))
    # sample the ends exactly: where E_N stops existing (the top end meets the
    # limiting inverse deflagrations), then where alpha_N does
    has_eN = lambda r: bool(np.all(np.isfinite(r)))
    eC_jouguet, results = _refine_edges(jouguet, eC_jouguet, results, has_eN)
    keep = np.array([has_eN(r) for r in results], dtype=bool)
    eC_jouguet, results = _refine_edges(jouguet, eC_jouguet[keep], [r for r, k in zip(results, keep) if k],
                                        lambda r: has_eN(r) and bool(np.isfinite(_alphan(r[1]))))
    j_xiw = np.array([r[0] for r in results], dtype=float)
    j_eN  = np.array([r[1] for r in results], dtype=float)
    j_eC  = np.asarray(eC_jouguet, dtype=float)
    ax_det.plot(j_xiw, j_eN, '--b', lw=2)
    ax_C.plot(j_xiw, j_eC, '--b', lw=2)
    ax_alpha.plot(j_xiw, np.array(list(map(_alphan, j_eN))), '--b', lw=2)
    results_dict['Local_contours']['inverse_jouguet'] = {
        'xi': j_xiw, 'eC': j_eC, 'eN': j_eN,
        'alpha': np.array(list(map(_alphan, j_eN)))}

    # Limiting inverse deflagrations: the rarefaction reaches rest above the
    # DCP with an infinite slope; the point of rest xi0 = c_s(E_N) is swept
    fig_det.suptitle("Computing limiting inverse deflagrations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    e_star = dcp_energy(outside)
    if np.isfinite(e_star):
        c_star = float(np.sqrt(outside.cs2(e_star)))
        xi0_lim = np.linspace(0.005 * c_star, 0.999 * c_star, contour_resolution * 10)
        limiting = lambda xi0: limiting_inverse_def_contour_finder(xi0, outside, inside, e_star=e_star)[:3]
        with concurrent.futures.ThreadPoolExecutor() as ex:
            results = list(ex.map(limiting, xi0_lim))
        # the end towards c_s(e*) is where the wall becomes Jouguet: the corner
        # shared with the inverse Jouguet curve
        xi0_lim, results = _refine_edges(limiting, xi0_lim, results, lambda r: bool(np.all(np.isfinite(r))))
        l_xiw, l_eN, l_eC = (np.array(col, dtype=float) for col in zip(*results))
        mask = np.isfinite(l_xiw) & np.isfinite(l_eN) & np.isfinite(l_eC)
        l_xiw, l_eN, l_eC = l_xiw[mask], l_eN[mask], l_eC[mask]
        ax_det.plot(l_xiw, l_eN, '--r', lw=2)
        ax_C.plot(l_xiw, l_eC, '--r', lw=2)
        ax_alpha.plot(l_xiw, np.array(list(map(_alphan, l_eN))), '--r', lw=2)
        results_dict['Local_contours']['limiting_inverse_deflagration'] = {
            'xi': l_xiw, 'eC': l_eC, 'eN': l_eN, 'e_star': e_star,
            'alpha': np.array(list(map(_alphan, l_eN)))}

    # Slowest inverse detonations: the state behind the wall at the inside
    # turning point; slower walls have no junction state on the inside branch.
    # Where that wall is not supersonic with respect to the outside phase the
    # curve is cut (NaN rows break the line)
    fig_det.suptitle("Computing slowest inverse detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    span = outside.e_hi - outside.e_lo
    en_range_det = (outside.e_lo + 1e-3 * span, outside.e_hi - 1e-3 * span)
    eN_spin = np.linspace(*en_range_det, contour_resolution * 10)
    spinodal = lambda eN: spinodal_inverse_det_contour_finder(eN, outside, inside)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(spinodal, eN_spin))
    eN_spin, results = _refine_edges(spinodal, eN_spin, results, lambda r: bool(np.all(np.isfinite(r))))
    s_xiw, s_eC = (np.array(col, dtype=float) for col in zip(*results))
    s_eN = np.asarray(eN_spin, dtype=float)
    s_alpha = np.array([_alphan(e) if np.isfinite(x) else np.nan for x, e in zip(s_xiw, s_eN)])
    ax_det.plot(s_xiw, s_eN, '--k')            # as the supercooled detonation separator (e- = eTL)
    ax_C.plot(s_xiw, s_eC, '--k')
    ax_alpha.plot(s_xiw, s_alpha, '--k')
    results_dict['Local_contours']['spinodal_inverse_detonation'] = {
        'xi': s_xiw, 'eC': s_eC, 'eN': s_eN, 'alpha': s_alpha}

    # Inverse detonations with the wall at the sound speed of the outside
    # phase, xi_w = c_s,out(E_N): the border with the inverse hybrids
    fig_det.suptitle("Computing sonic inverse detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    eN_sonic = np.linspace(*en_range_det, contour_resolution * 10)
    sonic = lambda eN: sonic_inverse_det_contour_finder(eN, outside, inside)
    with concurrent.futures.ThreadPoolExecutor() as ex:
        results = list(ex.map(sonic, eN_sonic))
    eN_sonic, results = _refine_edges(sonic, eN_sonic, results, lambda r: bool(np.all(np.isfinite(r))))
    c_xiw, c_eC = (np.array(col, dtype=float) for col in zip(*results))
    c_eN = np.asarray(eN_sonic, dtype=float)
    c_alpha = np.array([_alphan(e) if np.isfinite(x) else np.nan for x, e in zip(c_xiw, c_eN)])
    ax_det.plot(c_xiw, c_eN, '--k')            # as the supercooled deflagration separator, xi_w = c_s(E_C)
    ax_C.plot(c_xiw, c_eC, '--k')
    ax_alpha.plot(c_xiw, c_alpha, '--k')
    results_dict['Local_contours']['sonic_inverse_detonation'] = {
        'xi': c_xiw, 'eC': c_eC, 'eN': c_eN, 'alpha': c_alpha}

    fig_det.suptitle("Computing inverse deflagrations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()

    # ------------------------------------------------------------------
    # Progress bars
    # ------------------------------------------------------------------
    bh, bs, by = 0.015, 0.003, 0.94
    bars = {}
    for k, (key, label, colour) in enumerate((('def', "Inverse def.", 'blue'), ('det', "Inverse det.", 'red'))):
        ax_bar = fig_det.add_axes([0.15, by - k * (bh + bs), 0.7, bh])
        ax_bar.set_xticks([]);  ax_bar.set_yticks([])
        ax_bar.set_xlim(0, 1);  ax_bar.set_ylim(0, 1)
        ax_bar.text(-0.02, 0.5, label, va='center', ha='right', fontsize=8, transform=ax_bar.transAxes)
        bars[key] = (ax_bar, ax_bar.barh([0.5], [0], height=1.0, color=colour)[0],
                     ax_bar.text(1.02, 0.5, "0%", va='center', ha='left', fontsize=8,
                                 transform=ax_bar.transAxes))

    def _progress(key):
        def update(done, total):
            _, rect, pct = bars[key]
            rect.set_width(done / total)
            pct.set_text(f"{done / total * 100:.1f}%")
            fig_det.canvas.draw_idle();  fig_det.canvas.flush_events()
        return update

    # ------------------------------------------------------------------
    # Inverse deflagrations
    # ------------------------------------------------------------------
    rows = scan_inverse_deflagrations(outside, inside, xiw_range, ec_range,
                                      xiw_resolution, en_resolution, progress=_progress('def'))
    inverse_deflagration_flows = [(xiw, eN, eC, flow[0], flow[1], flow[2])
                                  for xiw, eC, eN, flow in rows]
    print(f"[snobex] {len(inverse_deflagration_flows)} inverse deflagrations")
    _batch_plot(ax_det, ax_C, ax_alpha, inverse_deflagration_flows, 'bo',
                'Inverse deflagration', 4, _alphan)

    # ------------------------------------------------------------------
    # Inverse detonations: supersonic walls, so xi_w up to 1, over the whole
    # outside branch in E_N
    # ------------------------------------------------------------------
    fig_det.suptitle("Computing inverse detonations…"); fig_det.canvas.draw(); fig_det.canvas.flush_events()
    rows = scan_inverse_detonations(outside, inside, (0.01, 0.99), en_range_det,
                                    xiw_resolution, en_resolution, progress=_progress('det'))
    for ax_bar, _, _ in bars.values():
        ax_bar.remove()
    fig_det.suptitle(None)
    inverse_detonation_flows = [(xiw, eN, eC, flow[0], flow[1], flow[2])
                                for xiw, eN, eC, flow in rows]
    print(f"[snobex] {len(inverse_detonation_flows)} inverse detonations")
    if inverse_detonation_flows:
        _batch_plot(ax_det, ax_C, ax_alpha, inverse_detonation_flows, 'ro',
                    'Inverse detonation', 4, _alphan)

    # ------------------------------------------------------------------
    # Allowed energies: superheated outside phase, inside phase above T_c
    # ------------------------------------------------------------------
    if e_out_Tc is not None:
        ax_det.axhspan(e_out_Tc, outside.e_turn, color='orange', alpha=0.2,
                       label=r'Allowed $\mathcal{E}_N$')
    if e_in_Tc is not None and (inverse_deflagration_flows or inverse_detonation_flows):
        eC_found = np.array([r[2] for r in inverse_deflagration_flows + inverse_detonation_flows])
        top = max(e_in_Tc, eC_found.max())
        ax_C.axhspan(e_in_Tc, top + 0.05 * (top - eC_found.min()), color='green', alpha=0.2,
                     label=r'Allowed $\mathcal{E}_C$')

    # ------------------------------------------------------------------
    # Entropy condition at the wall
    # ------------------------------------------------------------------
    # Delta J_s = J_s+ - J_s- >= 0, as entropy_checker for the supercooled walls
    jump = np.array([inverse_def_entropy_jump(r[2], r[0], r[3:6], outside, inside)
                     for r in inverse_deflagration_flows])
    jump_det = np.array([inverse_det_entropy_jump(r[1], r[0], r[3:6], outside, inside)
                         for r in inverse_detonation_flows])
    bad = ([r for r, x in zip(inverse_deflagration_flows, jump) if np.isfinite(x) and x < 0.0]
           + [r for r, x in zip(inverse_detonation_flows, jump_det) if np.isfinite(x) and x < 0.0])
    if bad:
        _batch_plot(ax_det, ax_C, ax_alpha, bad, 'k*', 'Entropy violated', 2.5, _alphan)
    results_dict['Hydro_flows']['inverse_deflagration'] = {
        **_pack_flows(inverse_deflagration_flows, _alphan),
        'entropy_jump': jump,
    }
    results_dict['Hydro_flows']['inverse_detonation'] = {
        **_pack_flows(inverse_detonation_flows, _alphan),
        'entropy_jump': jump_det,
    }

    # ------------------------------------------------------------------
    # Final plot decoration
    # ------------------------------------------------------------------
    for ax in (ax_det, ax_C, ax_alpha):
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    ax_det.set_title(
        rf"Superheated Solutions for EoS $e_{{TL}}-e_{{TH}} = {outside.e_turn - inside.e_turn:.2f}$"
        + "\n(Click on points to see flow details)"
    )
    # Inverse deflagrations occupy small xi_w: zoom the E_N panel like the others
    ax_det.relim();  ax_det.autoscale(enable=True, axis='x')

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    if saving:
        save_path = os.path.join(
            output_dir,
            f"snobex_results_superheated_{inside.e_turn:.3f}_{outside.e_turn:.3f}.pkl",
        )
        with open(save_path, 'wb') as f:
            pickle.dump(results_dict, f)
        print(f"[snobex] Results saved to '{save_path}'")

    # ------------------------------------------------------------------
    # Click handler
    # ------------------------------------------------------------------
    fig_det.canvas.mpl_connect(
        'button_press_event',
        lambda ev: _on_click(ev, ax_det, ax_C, ax_alpha,
                             inverse_detonation_flows, inverse_deflagration_flows, [], [],
                             output_dir, _alphan,
                             deflagration_label='Inverse deflagration',
                             detonation_label='Inverse detonation'),
    )

    plt.show()
    return fig_det, ax_det


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_txt(output_dir, filename, data):
    path = os.path.join(output_dir, filename)
    try:
        np.savetxt(path, data)
    except OSError:
        pass   # non-fatal if directory is read-only


def _plot_eos(pplus, pminus, eTH, eTL, p0, p1, e_max):
    e_vals = np.linspace(0, e_max, 500)
    fig_eos, ax_eos = plt.subplots(figsize=(8, 5))
    ax_eos.plot(e_vals, pplus(e_vals, eTH, p1), label='p+', color='tab:blue', lw=2)
    ax_eos.plot(e_vals, pminus(e_vals, eTL, p0), label='p−', color='tab:orange', lw=2)
    ax_eos.set_xlabel(r'$\mathcal{E}$')
    ax_eos.set_ylabel(r'$\mathcal{P}(\mathcal{E})$')
    ax_eos.set_title('Equation of State')
    ax_eos.legend();  ax_eos.grid(True)
    fig_eos.tight_layout();  fig_eos.show()


def _refine_edges(f, xs, rows, valid, n_iter=12):
    """
    Add samples of a curve x -> f(x) at the ends of its valid stretches.
    Between every pair of neighbouring samples whose ``valid(f(x))``
    differs, bisect towards the edge and keep the last valid point.

    Returns (xs, rows) with the new points merged in, sorted by x.
    """
    xs, rows = list(np.asarray(xs, dtype=float)), list(rows)
    ok = [valid(r) for r in rows]
    for i in range(len(ok) - 1):
        if ok[i] == ok[i + 1]:
            continue
        good, bad = (xs[i], xs[i + 1]) if ok[i] else (xs[i + 1], xs[i])
        best = None
        for _ in range(n_iter):
            mid = 0.5 * (good + bad)
            r = f(mid)
            if valid(r):
                good, best = mid, r
            else:
                bad = mid
        if best is not None:
            xs.append(good)
            rows.append(best)
    order = np.argsort(xs)
    return np.asarray(xs)[order], [rows[k] for k in order]


def _batch_plot(ax_det, ax_C, ax_alpha, flows, marker, label, ms, alphan_func):
    xiws = [r[0] for r in flows]
    es   = [r[1] for r in flows]
    eCs  = [r[2] for r in flows]
    ax_det.plot(xiws, es,   marker, label=label, ms=ms)
    ax_C.plot(xiws,   eCs,  marker, label=label, ms=ms)
    ax_alpha.plot(xiws, np.array(list(map(alphan_func, es))), marker, label=label, ms=ms)


def _pack_flows(flows, alphan_func):
    xiws = np.array([r[0] for r in flows])
    es   = np.array([r[1] for r in flows])
    eCs  = np.array([r[2] for r in flows])
    return {
        'xi': xiws, 'eC': eCs, 'eN': es,
        'alpha': np.array(list(map(alphan_func, es))),
        'flows': flows,
    }


def _entropy_overlay(ax_det, ax_C, ax_alpha, flows, bad_arr, marker, ms, alphan_func):
    if len(bad_arr) == 0:
        return
    xiws = np.array([r[0] for r in flows])
    es   = np.array([r[1] for r in flows])
    eCs  = np.array([r[2] for r in flows])
    sel  = bad_arr < 0
    if not np.any(sel):
        return
    ax_det.plot(xiws[sel], es[sel],   marker, ms=ms)
    ax_C.plot(xiws[sel],   eCs[sel],  marker, ms=ms)
    ax_alpha.plot(xiws[sel], np.array(list(map(alphan_func, es[sel]))), marker, ms=ms)


def _on_click(event, ax_det, ax_C, ax_alpha,
              detonation_data, deflagration_data, hybrid_data, shocked_data,
              output_dir, alphan_func, deflagration_label='Deflagration',
              detonation_label='Detonation'):
    if event.inaxes not in (ax_det, ax_C, ax_alpha):
        return
    xiw_click = event.xdata
    ey_click  = event.ydata
    if xiw_click is None or ey_click is None:
        return

    is_eN_plot    = (event.inaxes == ax_det)
    is_eC_plot    = (event.inaxes == ax_C)
    is_alpha_plot = (event.inaxes == ax_alpha)
    y_name = 'E_N' if is_eN_plot else ('E_C' if is_eC_plot else 'Alpha')
    print(f"[snobex] Click at ξ_w={xiw_click:.3f}, {y_name}={ey_click:.3f}")

    best_dist = np.inf
    best_sol  = None
    best_type = None

    for row in detonation_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row + (np.nan,);  best_type = detonation_label

    for row in deflagration_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row + (np.nan,);  best_type = deflagration_label

    for row in hybrid_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row;  best_type = 'Hybrid'

    for row in shocked_data:
        xiw, eN, eC = row[0], row[1], row[2]
        if is_eN_plot:
            ey = eN
        elif is_eC_plot:
            ey = eC
        else:
            ey = alphan_func(eN)
        d = np.hypot(xiw - xiw_click, ey - ey_click)
        if d < best_dist:
            best_dist = d;  best_sol = row;  best_type = 'Shocked det.'

    if best_sol is None:
        return

    xiw_s, eN_s, eC_s, v_f, xi_f, e_f = best_sol[:6]
    print(f"[snobex] → {best_type}: ξ_w={xiw_s:.3f}, E_N={eN_s:.3f}, E_C={eC_s:.3f}")

    y_label = 'E_N' if is_eN_plot else ('E_C' if is_eC_plot else 'Alpha')
    y_val   = eN_s  if is_eN_plot else (eC_s if is_eC_plot else alphan_func(eN_s))

    fig_f, (ax_v, ax_e) = plt.subplots(1, 2, figsize=(12, 5))
    ax_v.plot(xi_f, v_f, 'b-', lw=2, label=best_type)
    ax_v.set_xlabel(r'$\xi$');  ax_v.set_ylabel(r'$v(\xi)$')
    ax_v.set_title(f"Velocity — {best_type}\nξ_w={xiw_s:.3f}, {y_label}={y_val:.3f}")
    ax_v.grid(True);  ax_v.legend()

    ax_e.plot(xi_f, e_f, 'r-', lw=2, label=best_type)
    ax_e.set_xlabel(r'$\xi$');  ax_e.set_ylabel(r'$e(\xi)$')
    ax_e.set_title(f"Energy — {best_type}\nξ_w={xiw_s:.3f}, {y_label}={y_val:.3f}")
    ax_e.grid(True);  ax_e.legend()

    plt.tight_layout();  plt.show()

    _save_txt(output_dir, "vofxi.txt", np.stack((xi_f, v_f), axis=1))
    _save_txt(output_dir, "eofxi.txt", np.stack((xi_f, e_f), axis=1))
