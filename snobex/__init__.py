def print_logo():
    """Print the snobex ASCII logo to the terminal."""
    print(r"""
    _____ _   _  ____  ____  ________   __
  / ____| \ | |/ __ \|  _ \|  ____\ \ / /
 | (___ |  \| | |  | | |_) | |__   \ V /
  \___ \| . ` | |  | |  _ <|  __|   > <
  ____) | |\  | |__| | |_) | |____ / . \
 |_____/|_| \_|\____/|____/|______/_/ \_\
    """)


from .eos import (
    Branch,
    pplus,
    pminus,
    speed_of_sound_squared,
    load_custom_eos,
    analytic_branches,
    load_custom_branches,
    branches_from_custom_eos,
)
from .thermodynamics import (
    entropy_ODE_plus,
    entropy_ODE_minus,
    calculate_temperature,
    find_critical_temperature,
)
from .supercooled import (
    find_deto,
    find_def,
    find_hyb,
    find_shocked_detonation,
    find_shocked_detonation_jouguet,
)
from .plotting import interactive_bubble_plot, superheated_bubble_plot

__all__ = [
    "Branch",
    "pplus",
    "pminus",
    "speed_of_sound_squared",
    "load_custom_eos",
    "analytic_branches",
    "load_custom_branches",
    "branches_from_custom_eos",
    "print_logo",
    "entropy_ODE_plus",
    "entropy_ODE_minus",
    "calculate_temperature",
    "find_critical_temperature",
    "find_deto",
    "find_def",
    "find_hyb",
    "find_shocked_detonation",
    "find_shocked_detonation_jouguet",
    "interactive_bubble_plot",
    "superheated_bubble_plot",
]
