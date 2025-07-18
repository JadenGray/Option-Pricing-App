# src/minute_simulator.py
import pandas as pd
import numpy as np
from datetime import datetime
from black_scholes import black_scholes


def simulate_minute_chain(
    underlying_df: pd.DataFrame,
    option_chain: pd.DataFrame,
    r: float = 0.0,
    vol_surface: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Simulate minute-level option bid/ask prices by repricing with Black-Scholes.

    Parameters:
    - underlying_df: DataFrame with ['datetime', 'underlying_price']
    - option_chain: static DataFrame of options with ['option_symbol','strike','expiry','option_type']
    - r: risk-free rate
    - vol_surface: DataFrame with implied vols per strike/expiry (optional)

    Returns:
    - DataFrame with columns ['datetime','option_symbol','strike','expiry',
      'option_type','underlying_price','bid','ask','mid']
    """
    # Prepare base chain
    chain = option_chain.copy()
    chain = chain[['option_symbol','strike','expiry','option_type']].drop_duplicates()

    # Broadcast chain across each minute snapshot
    u = underlying_df.copy()
    u['key'] = 1
    chain['key'] = 1
    sim = u.merge(chain, on='key').drop('key', axis=1)

    # Compute time to expiry T
    sim['T'] = (pd.to_datetime(sim['expiry']) - sim['datetime']).dt.total_seconds() / (365 * 24 * 3600)
    sim = sim[sim['T'] > 0]

    # Determine sigma from vol_surface or flat
    if vol_surface is not None:
        # merge on strike and expiry
        sim = sim.merge(vol_surface, on=['strike','expiry'], how='left')
        sim['sigma'].fillna(sim['sigma'].mean(), inplace=True)
    else:
        sim['sigma'] = vol_surface['sigma'].mean() if hasattr(vol_surface, 'sigma') else 0.2

    # Reprice: mid = BS price
    sim['mid'] = sim.apply(
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

    # Set bid/ask around mid with 1% spread
    sim['bid'] = sim['mid'] * 0.995
    sim['ask'] = sim['mid'] * 1.005

    return sim.reset_index(drop=True)