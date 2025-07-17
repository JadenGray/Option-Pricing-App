# Makes src a package and exposes core functionality for easy imports
from .data_loader import load_options_data
from .black_scholes import black_scholes_metric, implied_volatility
from .analytics import (
    portfolio_metric, generate_heatmap_grid,
    plot_heatmap, plot_pnl_curve
)
from .strategy import backtest_example, analysis_example

__all__ = [
    'load_options_data', 'black_scholes_metric', 'implied_volatility',
    'portfolio_metric', 'generate_heatmap_grid', 'plot_heatmap',
    'plot_pnl_curve', 'backtest_example', 'analysis_example'
]