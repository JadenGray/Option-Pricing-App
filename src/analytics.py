def compute_pnl(df):
    df = df.sort_values(["option_type", "strike", "expiry", "date"])
    df["next_mid"] = df.groupby(["option_type", "strike", "expiry"])["mid"].shift(-1)
    df = df.dropna(subset=["next_mid"])
    df["pnl"] = (df["next_mid"] - df["mid"]) * df["signal"] * 100
    summary = df.groupby("date")["pnl"].sum().reset_index()
    summary["cumulative_pnl"] = summary["pnl"].cumsum()
    return summary