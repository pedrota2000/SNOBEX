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


from .eos import pplus, pminus, speed_of_sound_squared, load_custom_eos
from .thermodynamics import (
    entropy_ODE_plus,
    entropy_ODE_minus,
    calculate_temperature,
    find_critical_temperature,
)
from .hydrodynamics import find_deto, find_def, find_hyb
from .plotting import interactive_bubble_plot

__all__ = [
    "pplus",
    "pminus",
    "speed_of_sound_squared",
    "load_custom_eos",
    "print_logo",
    "entropy_ODE_plus",
    "entropy_ODE_minus",
    "calculate_temperature",
    "find_critical_temperature",
    "find_deto",
    "find_def",
    "find_hyb",
    "interactive_bubble_plot",
]
