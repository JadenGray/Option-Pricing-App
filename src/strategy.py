from .data_loader import load_options_data
from .analytics import portfolio_metric, generate_heatmap_grid, plot_heatmap, plot_pnl_curve
import optopsy as op


def backtest_example(csv_path, r, sigma):
    df = load_options_data(csv_path)
    stats = op.long_calls(df, moneyness=(0.99,1.01), days_to_expiry=(5,30))
    return stats


def analysis_example(portfolio, r, sigma):
    # define ranges
    S_range = (50,150)
    sigma_range = (0.1,0.5)
    # heatmap
    X,Y,Z = generate_heatmap_grid(portfolio, r, sigma, S_range, sigma_range, 'PnL')
    hm_fig = plot_heatmap(X,Y,Z,'PnL')
    # pnl curve
    pc_fig = plot_pnl_curve(portfolio, r, sigma, S_range)
    return hm_fig, pc_fig