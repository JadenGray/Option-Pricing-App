def compute_pnl(df):
    # sort by contract and timestamp
    df = df.sort_values(['option_symbol','datetime'])
    df['next_exec_price'] = df.groupby('option_symbol')['exec_price'].shift(-1)
    # PnL = (next_exec_price - exec_price) * position * 100
    df['pnl'] = (df['next_exec_price'] - df['exec_price']) * df['signal'] * 100
    daily = df.groupby(df['datetime'].dt.date)['pnl'].sum().reset_index(name='daily_pnl')
    daily['cumulative_pnl'] = daily['daily_pnl'].cumsum()
    return daily