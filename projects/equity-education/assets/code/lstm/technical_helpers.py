import numpy as np


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_support_resistance(prices, window=20, num_levels=3):
    series = prices.dropna()
    if series.empty:
        return [], []

    n = len(series)
    dyn_window = min(window, max(5, n // 2))
    if dyn_window % 2 == 0:
        dyn_window -= 1
    dyn_window = max(3, dyn_window)

    highs = series.rolling(
        dyn_window, center=True, min_periods=max(3, dyn_window // 2)
    ).max()
    lows = series.rolling(
        dyn_window, center=True, min_periods=max(3, dyn_window // 2)
    ).min()
    resist_pts = series[series >= highs].dropna()
    support_pts = series[series <= lows].dropna()

    if resist_pts.empty:
        neighbor_peaks = (series.shift(1) < series) & (series.shift(-1) <= series)
        resist_pts = series[neighbor_peaks].dropna()
    if support_pts.empty:
        neighbor_troughs = (series.shift(1) > series) & (series.shift(-1) >= series)
        support_pts = series[neighbor_troughs].dropna()

    if resist_pts.empty:
        q_hi = [0.80, 0.90, 0.95][:max(1, num_levels)]
        resist_pts = series.quantile(q_hi)
    if support_pts.empty:
        q_lo = [0.20, 0.10, 0.05][:max(1, num_levels)]
        support_pts = series.quantile(q_lo)

    def cluster(series_values, levels):
        if series_values.empty:
            return []
        vals = np.array(series_values.values, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return []
        vals = np.unique(vals)
        k = min(levels, len(vals))
        if k == 0:
            return []
        if k == 1:
            return [float(vals.mean())]
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(vals.reshape(-1, 1))
        return sorted(km.cluster_centers_.flatten())

    return cluster(support_pts, num_levels), cluster(resist_pts, num_levels)


def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i - lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


FEATURE_DISPLAY = {
    "RSI": "Relative Strength Index (RSI)",
    "MACD_Hist": "MACD Histogram",
    "MA5_Ratio": "Price / 5-day MA",
    "MA20_Ratio": "Price / 20-day MA",
    "MA50_Ratio": "Price / 50-day MA",
    "Volatility": "10-day Volatility",
    "Pct_Change": "Daily % Change",
    "Ret_5d": "5-day Return",
    "Ret_20d": "20-day Return",
    "Volume_Ratio": "Volume Ratio (vs 20-day avg)",
    "Range_Pos_3mo": "3-Month Range Position",
}
