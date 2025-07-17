# src/strategy.py

import numpy as np
from src.black_scholes import black_scholes

def generate_signals(df, r=0.0, sigma=0.2):
    """
    Generates long/short signals based on Black–Scholes mispricing:
      - Long if market mid < model price
      - Short if market mid > model price

    Params:
    - df: DataFrame with columns ['mid', 'strike', 'expiry', 'option_type', 'date']
    - r: risk-free rate
    - sigma: assumed volatility

    Returns:
    - df with added 'bs_price' and 'signal' columns (invalid rows dropped)
    """

    # 1) Filter to valid numeric and maturities
    df = df[
        (df['mid'] > 0) &
        (df['strike'] > 0) &
        (df['expiry'] > df['date'])
    ].copy()

    # 2) Compute BS model price safely
    def safe_bs(row):
        try:
            return black_scholes(
                S=row['mid'],
                K=row['strike'],
                expiry=row['expiry'],
                r=r,
                sigma=sigma,
                option_type=row['option_type']
            )
        except Exception:
            return np.nan

    df['bs_price'] = df.apply(safe_bs, axis=1)

    # 3) Drop rows where BS failed
    df = df.dropna(subset=['bs_price'])

    # 4) Generate signals:  1 = long, -1 = short
    df['signal'] = np.where(df['mid'] < df['bs_price'], 1, -1)

    return df
