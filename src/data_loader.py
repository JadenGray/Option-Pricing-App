# src/data_loader.py

import pandas as pd
import yfinance as yf
from datetime import date

def fetch_options(symbol: str) -> pd.DataFrame:
    """
    Fetch the full current options chain for `symbol` using yfinance.
    
    Returns a DataFrame with columns:
      - date: today’s date (datetime.date)
      - underlying_price: current spot price
      - option_symbol: OPRA-style ticker
      - option_type: 'call' or 'put'
      - strike: strike price (float)
      - expiry: expiration date (datetime.date)
      - bid, ask, mid (floats)
    """
    tk = yf.Ticker(symbol)
    
    # Spot price (last close)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"No history found for {symbol}")
    underlying_price = hist["Close"].iloc[-1]
    today = date.today()
    
    all_options = []
    for exp in tk.options:
        # Convert expiry string once to a date
        expiry_date = pd.to_datetime(exp).date()
        
        chain = tk.option_chain(exp)
        for df_side, typ in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df_side.copy()
            df["option_type"] = typ
            df["expiry"] = expiry_date
            df["date"] = today
            df["underlying_price"] = underlying_price
            # Calculate mid price
            df["mid"] = (df["bid"] + df["ask"]) / 2
            
            # Keep only the relevant columns and rename contractSymbol
            sub = df[
                [
                    "date",
                    "underlying_price",
                    "contractSymbol",
                    "option_type",
                    "strike",
                    "expiry",
                    "bid",
                    "ask",
                    "mid",
                ]
            ].rename(columns={"contractSymbol": "option_symbol"})
            all_options.append(sub)
    
    if not all_options:
        return pd.DataFrame()
    
    result = pd.concat(all_options, ignore_index=True, sort=False)
    return result
