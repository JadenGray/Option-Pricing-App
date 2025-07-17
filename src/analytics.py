import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from .black_scholes import black_scholes_metric


def portfolio_metric(portfolio: list, S, r, sigma, quantity_type='Price') -> float:
    """Sum metric across position list."""
    total = 0
    for opt in portfolio:
        val = black_scholes_metric(
            S, opt['strike'], opt['expiry'], r, sigma, opt['type'], quantity_type
        )
        total += opt['quantity'] * val
    return total


def generate_heatmap_grid(portfolio, r, sigma, S_range, sigma_range, metric):
    """Return meshgrid and metric matrix for heatmap."""
    spots = np.linspace(*S_range, 50)
    vols  = np.linspace(*sigma_range, 50)
    X, Y = np.meshgrid(spots, vols)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = portfolio_metric(
                portfolio, X[i,j], r, Y[i,j], quantity_type=metric
            ) - (portfolio_metric(portfolio, X[i,j], r, Y[i,j], 'Price')
                 if metric=='PnL' else 0)
    return X, Y, Z


def plot_heatmap(X, Y, Z, metric):
    norm = TwoSlopeNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z)) if metric=='PnL' else None
    cmap = 'RdYlGn' if metric=='PnL' else 'coolwarm'
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(Z, cmap=cmap, norm=norm,
                xticklabels=np.round(X[0],2), yticklabels=np.round(Y[:,0],2),
                cbar_kws={'label': metric})
    ax.set_xlabel('Spot Price'); ax.set_ylabel('Volatility')
    plt.title(f'{metric} Heatmap')
    return fig


def plot_pnl_curve(portfolio, r, sigma, S_range):
    spots = np.linspace(*S_range, 100)
    pnl = [portfolio_metric(portfolio, S, r, sigma, 'Price') for S in spots]
    cost = sum(opt['quantity']*opt['purchase_price'] for opt in portfolio)
    pnl_net = np.array(pnl) - cost
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(spots, pnl_net)
    ax.axhline(0, linestyle='--')
    ax.set_xlabel('Spot Price'); ax.set_ylabel('PnL')
    plt.title('Portfolio PnL vs Spot')
    return fig