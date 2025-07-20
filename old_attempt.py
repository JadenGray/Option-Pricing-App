# single_option_test.py
import os
import pandas as pd
import requests
from datetime import datetime

# --- Configuration ---
API_KEY  = os.getenv('POLYGON_API_KEY', 'vbGmuUKSvRTv651AKaU6wweNTMAn2caB')
BASE_URL = 'https://api.polygon.io'

def polygon_get(path: str, params: dict = None) -> dict:
    params = (params.copy() if params else {})
    params['apiKey'] = API_KEY
    r = requests.get(f"{BASE_URL}{path}", params=params)
    r.raise_for_status()
    return r.json()

# === 1) Underlying minute bars for SPY ===
def fetch_underlying(symbol, start, end):
    path = f"/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"
    data = polygon_get(path, {'adjusted':'true','sort':'asc','limit':50000})
    df = pd.DataFrame(data.get('results', []))
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    df = df[['datetime','c']].rename(columns={'c':'price'})
    return df

# === 2) One option contract minute bars ===
def fetch_option_bars(option_ticker, date_str):
    path = f"/v2/aggs/ticker/{option_ticker}/range/1/minute/{date_str}/{date_str}"
    data = polygon_get(path, {'adjusted':'true','sort':'asc','limit':50000})
    df = pd.DataFrame(data.get('results', []))
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    return df[['datetime','o','h','l','c','v']].rename(
        columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}
    )

# ————— Run and save —————

# (1) Underlying
u = fetch_underlying('SPY', '2025-07-01', '2025-07-02')
print("Underlying head:")
print(u.head())
u.to_csv('underlying_SPY_1m.csv', index=False)

# (2) Single known option ticker
# e.g. SPY Jul 18 2025 615 Call: ticker format "O:SPY250718C00615000"
opt_ticker = "O:SPY250718C00615000"
date_str = "2025-07-18"
opt = fetch_option_bars(opt_ticker, date_str)
print(f"\nOption {opt_ticker} head:")
print(opt.head())
opt.to_csv(f'option_{opt_ticker.replace(":", "_")}_{date_str}_1m.csv', index=False)

print("\nCSVs written:")
print(" • underlying_SPY_1m.csv")
print(f" • option_{opt_ticker.replace(':','_')}_{date_str}_1m.csv")