"""
src package initialization: expose core functions for easy imports
"""

from .data_loader import fetch_options
from .strategy import generate_signals
from .analytics import compute_pnl
from .black_scholes import black_scholes, implied_volatility

__all__ = [
    "fetch_options",
    "generate_signals",
    "compute_pnl",
    "black_scholes",
    "implied_volatility",
]