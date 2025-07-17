import pandas as pd
import optopsy as op
import requests
import io

def load_options_data(symbol="SPY") -> pd.DataFrame:
    """
    Downloads a public SPY options sample CSV, converts to optopsy DataFrame.
    """
    url = "https://huggingface.co/datasets/Subh775/SPY_options_chain/resolve/main/spy_sample-1.csv"
    r = requests.get(url)
    r.raise_for_status()
    raw = pd.read_csv(io.BytesIO(r.content), parse_dates=['date', 'expiry'])
    
    df = op.DataFrame(
        raw,
        datetime_col='date',
        underlying_price_col='underlying_price',
        option_symbol_col='contractSymbol',
        strike_col='strike',
        expiry_col='expiry',
        bid_col='bid',
        ask_col='ask'
    )
    return df