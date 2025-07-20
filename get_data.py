import os, time
import pandas as pd
import requests
from datetime import datetime, timedelta

API_KEY = "vbGmuUKSvRTv651AKaU6wweNTMAn2caB"
BASE = "https://api.polygon.io"

def polygon_get(path, params=None):
    p = params.copy() if params else {}
    p['apiKey'] = API_KEY
    r = requests.get(BASE + path, params=p)
    r.raise_for_status()
    return r.json()

# 1️⃣ Underlying minute bars
def fetch_underlying(symbol, start, end):
    path = f"/v2/aggs/ticker/{symbol}/range/1/minute/{start}/{end}"
    data = polygon_get(path, {'adjusted':'true','sort':'asc','limit':50000})
    df = pd.DataFrame(data.get('results', []))
    df['datetime'] = pd.to_datetime(df['t'], unit='ms')
    return df[['datetime','c']].rename(columns={'c':'price'})

# 2️⃣ Option chain snapshot
def fetch_option_chain(symbol, as_of):
    path = f"/v3/reference/options/contracts"
    ds = as_of.strftime('%Y-%m-%d')
    params = {
        'underlying_ticker': symbol,
        'expiration_date.gte': ds,
        'order': 'asc',
        'limit': 1000
    }
    data = polygon_get(path, params)
    return data['results']


# 3️⃣ Minute bars for options
def fetch_option_bars(option_symbols, as_of):
    ds = as_of.strftime('%Y-%m-%d')
    rows = []
    for sym in option_symbols[:4]:  # up to 4 minute-bar calls
        path = f"/v2/aggs/ticker/{sym}/range/1/minute/{ds}/{ds}"
        data = polygon_get(path, {'adjusted':'true','sort':'asc','limit':50000})
        for b in data.get('results', []):
            rows.append({
                'symbol': sym,
                'datetime': datetime.fromtimestamp(b['t']/1000),
                'o': b['o'], 'h': b['h'], 'l': b['l'], 'c': b['c'], 'v': b['v']
            })
        time.sleep(60/5)
    return pd.DataFrame(rows)

# ➡️ Get data for first week of July
symbol = 'SPY'
start, end = '2025-07-01','2025-07-02'
as_of = datetime(2025, 7, 2).date()

print("Underlying bars:")
u = fetch_underlying(symbol, start, end)
print(u.head())
u.to_csv('underlying.csv', index=False)

print("\nOption chain snapshot:")
chain = fetch_option_chain(symbol, as_of)
chain_df = pd.DataFrame(chain)
print(chain_df.head())
chain_df.to_csv('chain.csv', index=False)

opt_syms = chain_df['ticker'].tolist()
print(f"\nOption contract count: {len(opt_syms)}")

print("\nOption minute bars:")
opt_bars = fetch_option_bars(opt_syms, as_of)
print(opt_bars.head())
opt_bars.to_csv('option_bars.csv', index=False)

