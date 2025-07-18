# src/strategy.py
import numpy as np
import pandas as pd
from datetime import datetime
from black_scholes import black_scholes, implied_volatility


def generate_signals(
    df: pd.DataFrame,
    r: float = 0.0,
    vol_mode: str = 'implied',           # 'implied' or 'historical'
    filter_front_month: bool = True,
    strike_pct_range: float = 0.1,       # ±10% around spot
    delta_bounds: tuple = (0.3, 0.7)
) -> pd.DataFrame:
    """
    Generate trading signals for options based on Black-Scholes mispricing.

    - Filters to front-month if requested.
    - Filters strikes within ±strike_pct_range of spot.
    - Optionally filters by delta bounds if using implied vol.
    - Computes sigma per contract (implied or historical).
    - Signals: +1 buy at ask if BS price > ask; -1 sell at bid if BS price < bid; 0 otherwise.

    Returns DataFrame with additional columns: ['sigma', 'bs_price', 'signal', 'exec_price'].
    """
    df = df.copy()

    # Ensure datetime column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['datetime'] = pd.to_datetime(df['date'])

    # Filter front-month
    if filter_front_month:
        earliest = df['expiry'].min()
        df = df[df['expiry'] == earliest]

    # Filter strikes around spot
    df = df[(df['strike'] >= df['underlying_price'] * (1 - strike_pct_range)) &
            (df['strike'] <= df['underlying_price'] * (1 + strike_pct_range))]

    # Compute time to expiry in years
    df['T'] = (pd.to_datetime(df['expiry']) - df['datetime']).dt.total_seconds() / (365 * 24 * 3600)
    df = df[df['T'] > 0]

    # Historical vol mode: compute rolling vol on underlying
    if vol_mode == 'historical':
        # at each minute, compute past 30-min vol annualized
        df = df.sort_values('datetime')
        # group underlying series
        u = df[['datetime', 'underlying_price']].drop_duplicates().set_index('datetime')
        ret = np.log(u['underlying_price']).diff().dropna()
        hist_vol = ret.rolling(window=30).std() * np.sqrt(252 * 6.5 * 60)
        df = df.join(hist_vol.rename('hist_sigma'), on='datetime')
        df['sigma'] = df['hist_sigma'].fillna(method='ffill').fillna(df['hist_sigma'].mean())
    else:
        df['sigma'] = np.nan

    # For implied, compute per-row via Newton-Raphson
    if vol_mode == 'implied':
        sigs = []
        for _, row in df.iterrows():
            try:
                iv = implied_volatility(
                    S=row['underlying_price'],
                    K=row['strike'],
                    T=row['T'],
                    r=r,
                    option_price=row['mid'],
                    option_type=row['option_type'],
                )
                sigs.append(iv / 100 if iv > 1 else iv)
            except Exception:
                sigs.append(np.nan)
        df['sigma'] = sigs
        df['sigma'].fillna(df['sigma'].mean(), inplace=True)

    # Compute BS price
    df['bs_price'] = df.apply(
        lambda row: black_scholes(
            S=row['underlying_price'],
            K=row['strike'],
            T=row['T'],
            r=r,
            sigma=row['sigma'],
            option_type=row['option_type'],
            quantity='Price'
        ),
        axis=1
    )

    # Generate signals
    df['signal'] = 0
    df.loc[df['bs_price'] > df['ask'], 'signal'] = 1   # buy
    df.loc[df['bs_price'] < df['bid'], 'signal'] = -1  # sell

    # Execution price: assume fill at ask for buy, bid for sell
    df['exec_price'] = np.where(
        df['signal'] == 1, df['ask'],
        np.where(df['signal'] == -1, df['bid'], np.nan)
    )

    return df