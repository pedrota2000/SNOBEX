"""
main_bubble_solver.py
=====================
snobex — Standalone Bubble Phase-Space Solver

Directly launches the interactive bubble phase-space computation without
the EoS-explorer GUI.  Useful when EoS parameters are already known.

Usage
-----
    python main_bubble_solver.py

Configuration
-------------
Edit the parameter block below before running.  All parameters are
documented in ``snobex.plotting.interactive_bubble_plot``.

Custom EoS
----------
Set ``CUSTOM_EOS_FILE`` to a plain-text table of the full EoS curve with
columns for p and e, and optionally T and s, which are otherwise rebuilt
from p(e) (see ``snobex.eos.load_custom_eos``).  The table
is split into its HT and LT branches at the c_s² = 0 turning points, and
eTH, eTL, the energy ranges, S(E) and T(E) are taken from the data instead
of the analytic parameters below.
"""

from snobex.plotting import interactive_bubble_plot
from snobex.eos import load_custom_eos

# ---------------------------------------------------------------------------
# EoS parameters — analytic parametrisation
# ---------------------------------------------------------------------------
ETH   = 0.5    # HT turning-point energy
ETL   = 2.0    # LT turning-point energy
P1    = 0.0    # Pressure at eHT  (p+(eHT) = p1)
P0    = 0.0    # Pressure offset for LT branch (usually 0)
N     = 4.0    # LT branch shape exponent
DELTA = 1.0    # HT branch smoothing scale
STL   = 8.45   # LT entropy anchor
STH   = 2.5    # HT entropy anchor

# ---------------------------------------------------------------------------
# Energy ranges for phase-space sampling
# ---------------------------------------------------------------------------
NUCLEATION_ENERGY_RANGE = (ETH, ETH + 1.5)   # (E_N_min, E_N_max)
BUBBLE_ENERGY_RANGE     = (1e-6, ETL)         # (E_C_min, E_C_max)

# ---------------------------------------------------------------------------
# Resolution (increase for publication-quality results)
# ---------------------------------------------------------------------------
XIW_RESOLUTION     = 25   # Wall-velocity grid points
EN_RESOLUTION      = 25   # Energy grid points per axis
CONTOUR_RESOLUTION = 150  # Resolution for separator/contour lines

# ---------------------------------------------------------------------------
# Optional: path to a custom EoS file (set to None to use analytic model)
# ---------------------------------------------------------------------------
CUSTOM_EOS_FILE    = None          # e.g.  "Data_holo/thermo_phiM1.txt"
CUSTOM_EOS_COLUMNS = (0, 1, 2, 3)  # columns of (T, p, e, s); (0, 6, 7, 1) for thermo_phiM*.txt;
                                   # None for T and s rebuilds them from p(e), e.g. (None, 1, 0, None)
CUSTOM_EOS_IDX_MIN = 0             # rows to skip at the high-energy end of the table
CUSTOM_EOS_TC      = None          # T_c if known (1.0 for tables in units of T_c)
CUSTOM_EOS_CS2_FROM = 'pe'         # 'pe': cs2 = dp/de; 'Ts': cs2 = (s/T) dT/ds (holographic tables)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SAVE_RESULTS = False   # Set True to pickle the results
OUTPUT_DIR   = "."     # Directory for output files

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from snobex import print_logo
    print_logo()
    eos = dict(eTH=ETH, eTL=ETL,
               nucleation_energies_allowed=NUCLEATION_ENERGY_RANGE,
               bubble_energies_allowed=BUBBLE_ENERGY_RANGE)
    if CUSTOM_EOS_FILE is not None:
        eos = load_custom_eos(CUSTOM_EOS_FILE, columns=CUSTOM_EOS_COLUMNS,
                              idx_min=CUSTOM_EOS_IDX_MIN, Tc=CUSTOM_EOS_TC,
                              cs2_from=CUSTOM_EOS_CS2_FROM)

    fig, ax = interactive_bubble_plot(
        p1=P1,
        p0=P0,
        n=N,
        delta=DELTA,
        sTL=STL,
        sTH=STH,
        xiw_resolution=XIW_RESOLUTION,
        en_resolution=EN_RESOLUTION,
        contour_resolution=CONTOUR_RESOLUTION,
        show_eos=True,
        saving=SAVE_RESULTS,
        output_dir=OUTPUT_DIR,
        **eos,
    )
