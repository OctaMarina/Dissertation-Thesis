#!/usr/bin/env python3
"""
Pipeline: Binance aggTrades → Volume Bars → Microstructural & Structural Features
=================================================================================
Based on: Marcos López de Prado, "Advances in Financial Machine Learning"
          Chapters 17 (Structural Breaks), 18 (Entropy), 19 (Microstructure)

Input:  BTCUSDT-aggTrades-YYYY-MM.csv/.zip files in ./data/
Output: volume_bars_features.csv

Columns from aggTrades:
  agg_trade_id | price | quantity | first_trade_id | last_trade_id | timestamp(ms) | is_buyer_maker

is_buyer_maker interpretation (Binance):
  TRUE  → buyer was maker → taker is SELLER → sell-initiated  → bt = -1
  FALSE → seller was maker → taker is BUYER → buy-initiated   → bt = +1
"""

import os
import re
import sys
import glob
import zipfile
import time as _time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy import stats as sp_stats
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = "volume_bars_features.csv"

# Volume bar threshold (in BTC). Will be auto-computed if set to None.
# Recommendation from Prado: 1/50 of average daily volume.
VOLUME_THRESHOLD = 1162.0  # Set to e.g. 1000.0 to override auto-computation

# Chunk size for reading large CSV files (rows per chunk)
CHUNK_SIZE = 5_000_000

# Rolling window for microstructural features (in bars)
ROLL_WINDOW = 50

# SADF parameters
SADF_MIN_SL = 20       # minimum sample length (tau)
SADF_LAGS = 1          # number of lags in ADF specification
SADF_CONSTANT = "nc"   # 'nc' = constant only, 'ct' = constant + trend

# CUSUM lookback (max backward-shifting windows)
CUSUM_LOOKBACK = 200

# Entropy parameters
ENTROPY_WINDOW = 100    # window size for rolling entropy
ENTROPY_ENCODE_QUANTILES = 10  # letters for quantile encoding

# VPIN
VPIN_WINDOW = 50        # number of bars for VPIN averaging


# ============================================================================
# PART 1: FILE I/O
# ============================================================================

AGGTRADES_COLS = [
    "agg_trade_id", "price", "quantity",
    "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker",
]


def list_aggtrades_files(data_dir: str) -> List[str]:
    """Return aggTrades file paths sorted chronologically."""
    patterns = [
        os.path.join(data_dir, "BTCUSDT-aggTrades-*.csv"),
        os.path.join(data_dir, "BTCUSDT-aggTrades-*.zip"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    def sort_key(f):
        m = re.search(r"(\d{4})-(\d{2})", os.path.basename(f))
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 99)

    return sorted(files, key=sort_key)


def _open_csv(filepath: str):
    """Return an open file-like object for csv or zip."""
    if filepath.endswith(".zip"):
        zf = zipfile.ZipFile(filepath, "r")
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            zf.close()
            return None, None
        return zf.open(csv_names[0]), zf
    else:
        return open(filepath, "r"), None


def read_aggtrades_chunked(filepath: str, chunk_size: int = CHUNK_SIZE):
    fobj, zf = _open_csv(filepath)
    if fobj is None:
        return

    try:
        reader = pd.read_csv(
            fobj,
            header=None,
            names=AGGTRADES_COLS,
            dtype="string",
            chunksize=chunk_size,
            on_bad_lines="skip",
        )

        for chunk in reader:
            # Convert numeric columns safely
            chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
            chunk["quantity"] = pd.to_numeric(chunk["quantity"], errors="coerce")
            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")

            # Drop header/junk rows (price/timestamp must be numeric)
            chunk = chunk.dropna(subset=["price", "quantity", "timestamp"]).copy()

            # Timestamp ms -> datetime
            chunk["date_time"] = pd.to_datetime(
                chunk["timestamp"].astype("int64"), unit="ms", utc=True
            )

            # is_buyer_maker -> bool
            chunk["is_buyer_maker"] = (
                chunk["is_buyer_maker"].astype(str).str.strip().str.lower().isin(["true", "1"])
            )

            yield chunk[["date_time", "price", "quantity", "is_buyer_maker"]]
    finally:
        if zf is not None:
            zf.close()


def estimate_daily_volume(data_dir: str, sample_files: int = 3) -> float:
    """Estimate average daily volume (BTC) from the first few files."""
    files = list_aggtrades_files(data_dir)[:sample_files]
    total_vol = 0.0
    min_ts, max_ts = None, None

    for f in files:
        print(f"  Sampling {os.path.basename(f)} for threshold estimation...")
        for chunk in read_aggtrades_chunked(f, chunk_size=2_000_000):
            total_vol += chunk["quantity"].sum()
            ts = chunk["date_time"]
            cmin, cmax = ts.min(), ts.max()
            if min_ts is None or cmin < min_ts:
                min_ts = cmin
            if max_ts is None or cmax > max_ts:
                max_ts = cmax

    if min_ts is None or max_ts is None:
        return 1000.0  # fallback

    days = max((max_ts - min_ts).total_seconds() / 86400, 1.0)
    avg_daily = total_vol / days
    return avg_daily


# ============================================================================
# PART 2: VOLUME BAR BUILDER
# ============================================================================

class VolumeBarBuilder:
    """
    Builds volume bars from streaming aggTrades data.
    Each bar is emitted when cumulative volume reaches `threshold`.

    Tracks buy/sell volume using is_buyer_maker (not tick rule),
    since Binance provides direct aggressor side information.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.completed_bars: List[Dict] = []

        # Running state
        self._vol_offset = 0.0          # cumulative volume before current chunk
        self._partial: Optional[Dict] = None  # incomplete bar at chunk boundary
        self._global_ticks = 0

    @staticmethod
    def _agg_group(group_df) -> Dict:
        """Aggregate a group of trades into bar statistics."""
        is_buy = ~group_df["is_buyer_maker"].values
        qty = group_df["quantity"].values
        prc = group_df["price"].values
        return {
            "open_time": group_df["date_time"].iloc[0],
            "close_time": group_df["date_time"].iloc[-1],
            "open": prc[0],
            "high": prc.max(),
            "low": prc.min(),
            "close": prc[-1],
            "volume": qty.sum(),
            "buy_volume": qty[is_buy].sum() if is_buy.any() else 0.0,
            "sell_volume": qty[~is_buy].sum() if (~is_buy).any() else 0.0,
            "ticks": len(group_df),
            "dollar_value": (prc * qty).sum(),
        }

    @staticmethod
    def _merge_bars(old: Dict, new: Dict) -> Dict:
        """Merge a partial bar with new data."""
        return {
            "open_time": old["open_time"],
            "close_time": new["close_time"],
            "open": old["open"],
            "high": max(old["high"], new["high"]),
            "low": min(old["low"], new["low"]),
            "close": new["close"],
            "volume": old["volume"] + new["volume"],
            "buy_volume": old["buy_volume"] + new["buy_volume"],
            "sell_volume": old["sell_volume"] + new["sell_volume"],
            "ticks": old["ticks"] + new["ticks"],
            "dollar_value": old["dollar_value"] + new["dollar_value"],
        }

    def process_chunk(self, df: pd.DataFrame):
        """Process a chunk of trades, emitting complete volume bars."""
        if len(df) == 0:
            return

        qty = df["quantity"].values

        # Continuous cumulative volume across all chunks
        cum_vol = np.cumsum(qty) + self._vol_offset

        # Assign each trade to a bar index
        bar_ids = (cum_vol / self.threshold).astype(np.int64)

        # Update offset for next chunk
        self._vol_offset = cum_vol[-1]

        # Last bar index and whether it's complete
        last_bar_id = bar_ids[-1]
        last_bar_vol_end = (last_bar_id + 1) * self.threshold
        last_bar_complete = cum_vol[-1] >= last_bar_vol_end - 1e-10

        # Group by bar_id
        df = df.copy()
        df["_bar_id"] = bar_ids
        unique_bars = df["_bar_id"].unique()

        for bid in unique_bars:
            group = df[df["_bar_id"] == bid]
            bar_data = self._agg_group(group)

            # Merge with partial bar from previous chunk
            if bid == unique_bars[0] and self._partial is not None:
                bar_data = self._merge_bars(self._partial, bar_data)
                self._partial = None

            # Is this the last (possibly incomplete) bar?
            is_last = bid == last_bar_id
            if is_last and not last_bar_complete:
                self._partial = bar_data
            else:
                self._global_ticks += bar_data["ticks"]
                bar_data["tick_num"] = self._global_ticks
                self.completed_bars.append(bar_data)

    def finalize(self) -> pd.DataFrame:
        """Return completed bars as a DataFrame. Discard any trailing partial bar."""
        if not self.completed_bars:
            return pd.DataFrame()

        df = pd.DataFrame(self.completed_bars)
        df = df.rename(columns={
            "close_time": "date_time",
            "buy_volume": "cum_buy_volume",
            "sell_volume": "cum_sell_volume",
            "ticks": "cum_ticks",
            "dollar_value": "cum_dollar_value",
        })
        cols = [
            "date_time", "tick_num", "open", "high", "low", "close",
            "volume", "cum_buy_volume", "cum_sell_volume",
            "cum_ticks", "cum_dollar_value",
        ]
        # Keep open_time as auxiliary
        df["bar_open_time"] = [b["open_time"] for b in self.completed_bars]
        return df[[c for c in cols if c in df.columns] + ["bar_open_time"]]


def build_volume_bars(data_dir: str, threshold: float) -> pd.DataFrame:
    """Build volume bars from all aggTrades files in data_dir."""
    files = list_aggtrades_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No aggTrades files found in {data_dir}")

    print(f"\n{'='*60}")
    print(f"BUILDING VOLUME BARS  |  threshold = {threshold:.2f} BTC")
    print(f"Files: {len(files)}  |  {os.path.basename(files[0])} → {os.path.basename(files[-1])}")
    print(f"{'='*60}\n")

    builder = VolumeBarBuilder(threshold=threshold)

    for i, filepath in enumerate(files):
        fname = os.path.basename(filepath)
        t0 = _time.time()
        n_rows = 0

        for chunk in read_aggtrades_chunked(filepath, chunk_size=CHUNK_SIZE):
            builder.process_chunk(chunk)
            n_rows += len(chunk)

        elapsed = _time.time() - t0
        n_bars = len(builder.completed_bars)
        print(f"  [{i+1:3d}/{len(files)}] {fname:40s} "
              f"rows={n_rows:>12,}  bars_so_far={n_bars:>8,}  ({elapsed:.1f}s)")

    bars = builder.finalize()
    print(f"\n  Total volume bars: {len(bars):,}")
    return bars


# ============================================================================
# PART 3: FEATURE COMPUTATION
# ============================================================================

# ---------------------------------------------------------------------------
# 3.1  Order Flow Imbalance (OI) & VPIN
# ---------------------------------------------------------------------------

def compute_oi_vpin(bars: pd.DataFrame, vpin_window: int = VPIN_WINDOW) -> pd.DataFrame:
    """
    Order Flow Imbalance:  OI_τ = (VB - VS) / (VB + VS)
    VPIN (Easley et al. 2012):  VPIN = mean(|VB - VS|) / V  over n bars

    Using is_buyer_maker directly (not tick rule).
    """
    vb = bars["cum_buy_volume"].values
    vs = bars["cum_sell_volume"].values
    vol = bars["volume"].values

    # OI per bar: bounded in [-1, 1]
    bars["oi"] = np.where(vol > 0, (vb - vs) / vol, 0.0)

    # VPIN: rolling mean of |VB - VS| / V
    abs_imbalance = np.abs(vb - vs)
    bars["vpin"] = (
        pd.Series(abs_imbalance).rolling(vpin_window, min_periods=1).mean().values / vol
    )
    # Normalize: use mean bar volume in window as denominator
    # More faithful to Easley: VPIN = sum(|VB-VS|) / (n*V)
    # Since all bars have volume ≈ threshold, V ≈ threshold, so:
    bars["vpin"] = (
        pd.Series(abs_imbalance).rolling(vpin_window, min_periods=1).mean().values
        / pd.Series(vol).rolling(vpin_window, min_periods=1).mean().values
    )

    return bars


# ---------------------------------------------------------------------------
# 3.2  OI Serial Correlation
# ---------------------------------------------------------------------------

def compute_oi_autocorr(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """Rolling first-order autocorrelation of order flow imbalance."""
    bars["oi_autocorr"] = (
        bars["oi"].rolling(window, min_periods=window).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
    )
    return bars


# ---------------------------------------------------------------------------
# 3.3  Kyle's Lambda  (Ch. 19.4.1)
# ---------------------------------------------------------------------------

def _rolling_regression_tval(y: np.ndarray, x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling OLS of y on x (univariate), return t-value of slope.
    Prado notes t-values are more informative than mean estimates.
    """
    n = len(y)
    tvals = np.full(n, np.nan)

    for i in range(window, n):
        yy = y[i - window : i]
        xx = x[i - window : i]
        xx_dm = xx - xx.mean()
        yy_dm = yy - yy.mean()
        ss_xx = np.dot(xx_dm, xx_dm)
        if ss_xx < 1e-20:
            continue
        beta = np.dot(xx_dm, yy_dm) / ss_xx
        resid = yy - (yy.mean() + beta * xx_dm)
        ss_res = np.dot(resid, resid)
        se_beta_sq = ss_res / ((window - 2) * ss_xx)
        if se_beta_sq <= 0:
            continue
        tvals[i] = beta / np.sqrt(se_beta_sq)

    return tvals


def compute_kyle_lambda(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Kyle (1985):  Δp_t = λ (b_t V_t) + ε_t
    signed_volume = VB - VS  (net order flow per bar)
    """
    dp = bars["close"].diff().values
    signed_vol = (bars["cum_buy_volume"] - bars["cum_sell_volume"]).values

    bars["kyle_lambda_tval"] = _rolling_regression_tval(dp, signed_vol, window)
    return bars


# ---------------------------------------------------------------------------
# 3.4  Amihud's Lambda  (Ch. 19.4.2)
# ---------------------------------------------------------------------------

def compute_amihud_lambda(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Amihud (2002):  |Δlog(p̃)| = λ Σ(p·V) + ε
    """
    abs_log_ret = np.abs(np.diff(np.log(bars["close"].values), prepend=np.nan))
    abs_log_ret[0] = np.nan
    dollar_vol = bars["cum_dollar_value"].values

    bars["amihud_lambda_tval"] = _rolling_regression_tval(abs_log_ret, dollar_vol, window)
    return bars


# ---------------------------------------------------------------------------
# 3.5  Hasbrouck's Lambda  (Ch. 19.4.3)
# ---------------------------------------------------------------------------

def compute_hasbrouck_lambda(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Hasbrouck (2009):  Δlog(p̃) = λ Σ(b_t √(p·V)) + ε
    signed_sqrt_dollar = sign(OI) * √(dollar_volume)
    """
    log_ret = np.diff(np.log(bars["close"].values), prepend=np.nan)
    log_ret[0] = np.nan

    oi_sign = np.sign(bars["cum_buy_volume"].values - bars["cum_sell_volume"].values)
    sqrt_dollar = np.sqrt(bars["cum_dollar_value"].values)
    signed_sqrt_dollar = oi_sign * sqrt_dollar

    bars["hasbrouck_lambda_tval"] = _rolling_regression_tval(
        log_ret, signed_sqrt_dollar, window
    )
    return bars


# ---------------------------------------------------------------------------
# 3.6  Roll Spread  (Ch. 19.3.2)
# ---------------------------------------------------------------------------

def compute_roll_spread(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Roll (1984):  c = √max{0, -Cov(Δp_t, Δp_{t-1})}
    Effective half bid-ask spread.
    """
    dp = bars["close"].diff().values
    n = len(dp)
    spread = np.full(n, np.nan)

    for i in range(window + 1, n):
        dp_w = dp[i - window : i]
        dp_lag = dp[i - window - 1 : i - 1]
        # Remove NaN
        mask = ~(np.isnan(dp_w) | np.isnan(dp_lag))
        if mask.sum() < 3:
            continue
        cov = np.cov(dp_w[mask], dp_lag[mask])[0, 1]
        spread[i] = np.sqrt(max(0.0, -cov))

    bars["roll_spread"] = spread
    return bars


# ---------------------------------------------------------------------------
# 3.7  Parkinson Volatility  (Ch. 19.3.3)
# ---------------------------------------------------------------------------

def compute_parkinson_vol(bars: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """
    Parkinson (1980):  σ²_HL = E[(log(H/L))²] / (4 log 2)
    """
    hl = np.log(bars["high"].values / bars["low"].values)
    hl_sq = hl ** 2
    k1 = 4.0 * np.log(2.0)

    bars["parkinson_vol"] = np.sqrt(
        pd.Series(hl_sq).rolling(window, min_periods=1).mean().values / k1
    )
    return bars


# ---------------------------------------------------------------------------
# 3.8  Corwin-Schultz Spread  (Ch. 19.3.4)
# ---------------------------------------------------------------------------

def compute_corwin_schultz(bars: pd.DataFrame, sl: int = 1) -> pd.DataFrame:
    """
    Corwin & Schultz (2012): Bid-ask spread from high-low prices.
    S_t = 2(e^α - 1) / (1 + e^α)
    """
    high = bars["high"].values
    low = bars["low"].values
    n = len(high)

    # β: average of sum of squared log(H/L) over 2 consecutive bars
    hl_sq = np.log(high / low) ** 2
    beta_raw = np.full(n, np.nan)
    for i in range(1, n):
        beta_raw[i] = hl_sq[i] + hl_sq[i - 1]

    beta = pd.Series(beta_raw).rolling(sl, min_periods=1).mean().values

    # γ: log(H_{t-1,t} / L_{t-1,t})²  over 2-bar high/low
    gamma = np.full(n, np.nan)
    for i in range(1, n):
        h2 = max(high[i], high[i - 1])
        l2 = min(low[i], low[i - 1])
        if l2 > 0:
            gamma[i] = np.log(h2 / l2) ** 2

    # α
    den = 3.0 - 2.0 * np.sqrt(2.0)
    sqrt_beta = np.sqrt(np.maximum(beta, 0.0))
    sqrt_gamma = np.sqrt(np.maximum(gamma, 0.0))
    alpha = (np.sqrt(2.0) - 1.0) * sqrt_beta / den - sqrt_gamma / np.sqrt(den)
    alpha = np.maximum(alpha, 0.0)  # set negative to 0 per paper p.727

    # Spread
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    # Beckers-Parkinson volatility as byproduct
    k2 = np.sqrt(8.0 / np.pi)
    sigma = (np.sqrt(0.5) - 1.0) * sqrt_beta / (k2 * den)
    sigma += sqrt_gamma / np.sqrt(k2 ** 2 * den)
    sigma = np.maximum(sigma, 0.0)

    bars["corwin_schultz_spread"] = spread
    bars["corwin_schultz_vol"] = sigma
    return bars


# ---------------------------------------------------------------------------
# 3.9  Chu-Stinchcombe-White CUSUM  (Ch. 17.3.2)
# ---------------------------------------------------------------------------

def compute_csw_cusum(
    bars: pd.DataFrame, lookback: int = CUSUM_LOOKBACK
) -> pd.DataFrame:
    """
    CSW CUSUM on log-price levels (Homm & Breitung 2012).
    S_{n,t} = (y_t - y_n) / (σ̂_t √(t-n))
    S_t = sup_{n ∈ [max(1,t-lookback), t]} S_{n,t}
    Critical value: c_α = √(b_α + log(t-n)),  b_{0.05} = 4.6
    """
    log_p = np.log(bars["close"].values)
    n = len(log_p)

    # Precompute σ̂²_t = (t-1)^{-1} Σ_{i=2}^{t} (Δy_i)²
    dlog = np.diff(log_p, prepend=np.nan)
    dlog_sq = dlog ** 2

    cusum_stat = np.full(n, np.nan)
    cusum_exceeded = np.full(n, np.nan)  # S_t > c_alpha? (binary feature)

    b_alpha = 4.6  # for α=0.05

    for t in range(2, n):
        # σ̂²_t estimated from all Δy up to t
        sigma2 = np.nanmean(dlog_sq[1 : t + 1])
        if sigma2 <= 0:
            continue
        sigma = np.sqrt(sigma2)

        # Backward-shifting reference point n
        n_start = max(0, t - lookback)
        best_s = -np.inf

        for ref in range(n_start, t):
            gap = t - ref
            if gap < 2:
                continue
            s_nt = (log_p[t] - log_p[ref]) / (sigma * np.sqrt(gap))
            if s_nt > best_s:
                best_s = s_nt

        cusum_stat[t] = best_s

        # Critical value for the supremum
        max_gap = t - n_start
        if max_gap >= 2:
            c_alpha = np.sqrt(b_alpha + np.log(max_gap))
            cusum_exceeded[t] = 1.0 if best_s > c_alpha else 0.0

    bars["csw_cusum"] = cusum_stat
    bars["csw_cusum_signal"] = cusum_exceeded
    return bars


# ---------------------------------------------------------------------------
# 3.10  SADF  (Ch. 17.4.2)
# ---------------------------------------------------------------------------

def _lag_matrix(series: np.ndarray, lags: int) -> np.ndarray:
    """Create a matrix of lagged values."""
    n = len(series)
    out = np.full((n, lags + 1), np.nan)
    for lag in range(lags + 1):
        out[lag:, lag] = series[: n - lag]
    return out


def _get_yx_adf(log_prices: np.ndarray, lags: int, constant: str) -> Tuple:
    """
    Prepare y and X for ADF specification:
    Δy_t = α + β y_{t-1} + Σ γ_l Δy_{t-l} + ε_t
    """
    dlog = np.diff(log_prices)
    n = len(dlog)

    # Lagged differences
    lag_mat = _lag_matrix(dlog, lags)  # shape (n, lags+1)

    # First column of lag_mat is Δy_t (lag=0), rest are Δy_{t-1}, ..., Δy_{t-lags}
    # y = Δy_t, x = [y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}, const, ...]

    # Trim to valid rows (no NaN from lagging)
    start = lags
    y = dlog[start:]  # Δy_t
    T = len(y)

    # X: [y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}]
    x_parts = []
    # Lagged level: y_{t-1} = log_prices[start:-1] if we align correctly
    x_parts.append(log_prices[start : start + T])  # y_{t-1}

    for l in range(1, lags + 1):
        x_parts.append(dlog[start - l : start - l + T])

    X = np.column_stack(x_parts)

    # Add constant
    if constant in ("nc", "ct", "ctt"):
        X = np.column_stack([X, np.ones(T)])
    if constant in ("ct", "ctt"):
        X = np.column_stack([X, np.arange(T)])
    if constant == "ctt":
        X = np.column_stack([X, np.arange(T) ** 2])

    return y, X


def _get_betas(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS: return (bMean, bVar). Snippet 17.4."""
    xy = X.T @ y
    xx = X.T @ X
    try:
        xxinv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return np.full((X.shape[1], 1), np.nan), np.full((X.shape[1], X.shape[1]), np.nan)
    bMean = xxinv @ xy
    err = y - X @ bMean
    dof = max(X.shape[0] - X.shape[1], 1)
    bVar = (err @ err / dof) * xxinv
    return bMean, bVar


def _sadf_inner(log_prices: np.ndarray, min_sl: int, lags: int, constant: str) -> float:
    """
    SADF inner loop (Snippet 17.1 adapted).
    Computes SADF_t = sup_{t0} { ADF_{t0,t} } for a fixed endpoint.
    """
    y_full, x_full = _get_yx_adf(log_prices, lags, constant)
    T = len(y_full)

    if T < min_sl:
        return np.nan

    best_adf = -np.inf
    # Backwards expanding: try start points from 0 to T - min_sl
    for start in range(0, T - min_sl + 1):
        y_ = y_full[start:]
        x_ = x_full[start:]
        if len(y_) < lags + 2:
            continue
        bMean, bVar = _get_betas(y_, x_)
        if np.isnan(bMean[0]) or bVar[0, 0] <= 0:
            continue
        adf_stat = bMean[0] / np.sqrt(bVar[0, 0])
        if adf_stat > best_adf:
            best_adf = adf_stat

    return best_adf if best_adf > -1e10 else np.nan


def compute_sadf(
    bars: pd.DataFrame,
    min_sl: int = SADF_MIN_SL,
    lags: int = SADF_LAGS,
    constant: str = SADF_CONSTANT,
    step: int = 1,
) -> pd.DataFrame:
    """
    Compute SADF series on log-prices from volume bars.
    Uses log prices as recommended in Ch. 17.4.2.1.

    step > 1 computes SADF every `step` bars (for speed) and forward-fills.
    """
    log_p = np.log(bars["close"].values)
    n = len(log_p)
    sadf_vals = np.full(n, np.nan)

    # Minimum bars needed: min_sl + lags + 1
    start_idx = min_sl + lags + 1

    print(f"  Computing SADF (n={n}, min_sl={min_sl}, step={step})...")

    for t in range(start_idx, n, step):
        sadf_vals[t] = _sadf_inner(log_p[: t + 1], min_sl, lags, constant)
        if t % 500 == 0:
            print(f"    SADF progress: {t}/{n}  ({100*t/n:.1f}%)")

    # Forward-fill if step > 1
    if step > 1:
        last_val = np.nan
        for i in range(n):
            if not np.isnan(sadf_vals[i]):
                last_val = sadf_vals[i]
            else:
                sadf_vals[i] = last_val

    bars["sadf"] = sadf_vals
    return bars


# ---------------------------------------------------------------------------
# 3.11  Lempel-Ziv / Kontoyiannis Entropy  (Ch. 18.4)
# ---------------------------------------------------------------------------

def _match_length(msg: str, i: int, n: int) -> int:
    """
    Maximum matched length + 1, with overlap.
    Adapted from Snippet 18.3.
    i >= n and len(msg) >= i + n
    """
    max_l = 0
    for l in range(n):
        pattern = msg[i : i + l + 1]
        found = False
        for j in range(i - n, i):
            if j + l + 1 > len(msg):
                break
            candidate = msg[j : j + l + 1]
            if pattern == candidate:
                found = True
                break
        if found:
            max_l = l + 1
        else:
            break
    return max_l + 1


def _konto_entropy(msg: str, window: Optional[int] = None) -> float:
    """
    Kontoyiannis' LZ entropy estimate (Snippet 18.4 adapted to Python 3).
    Uses the modified estimator to avoid the Doeblin condition.
    Returns entropy rate H̃.
    """
    if len(msg) < 4:
        return np.nan

    total = 0.0
    count = 0

    if window is None:
        # Expanding window
        if len(msg) % 2 != 0:
            msg = msg[1:]  # remove first bit for even length
        for i in range(1, len(msg) // 2 + 1):
            l = _match_length(msg, i, i)
            total += np.log2(i + 1) / l
            count += 1
    else:
        window = min(window, len(msg) // 2)
        for i in range(window, len(msg) - window + 1):
            l = _match_length(msg, i, window)
            total += np.log2(window + 1) / l
            count += 1

    if count == 0:
        return np.nan
    return total / count


def compute_lz_entropy(
    bars: pd.DataFrame,
    window: int = ENTROPY_WINDOW,
    encode_window: Optional[int] = 50,
) -> pd.DataFrame:
    """
    Rolling Lempel-Ziv (Kontoyiannis) entropy on binary-encoded returns.
    Encoding: 1 if return > 0, 0 if return <= 0.
    """
    log_ret = np.diff(np.log(bars["close"].values), prepend=np.nan)
    n = len(log_ret)

    # Binary encoding of return signs
    encoded = np.where(log_ret > 0, "1", "0")
    encoded[0] = "0"  # first return is NaN

    lz_vals = np.full(n, np.nan)

    print(f"  Computing LZ entropy (rolling window={window})...")
    for t in range(window, n):
        msg = "".join(encoded[t - window : t])
        lz_vals[t] = _konto_entropy(msg, encode_window)
        if t % 2000 == 0:
            print(f"    LZ entropy progress: {t}/{n}  ({100*t/n:.1f}%)")

    bars["lz_entropy"] = lz_vals
    return bars


# ---------------------------------------------------------------------------
# 3.12  Plug-In Entropy Estimator  (Ch. 18.3)
# ---------------------------------------------------------------------------

def _plug_in_entropy(msg: str, w: int) -> float:
    """
    Plug-in (ML) entropy rate estimator (Snippet 18.1 adapted).
    w = word length.
    """
    if len(msg) <= w:
        return np.nan

    # Build PMF
    lib = {}
    for i in range(w, len(msg)):
        word = msg[i - w : i]
        lib[word] = lib.get(word, 0) + 1

    total = len(msg) - w
    if total <= 0:
        return np.nan

    pmf = {k: v / total for k, v in lib.items()}
    h = -sum(p * np.log2(p) for p in pmf.values()) / w
    return h


def compute_plugin_entropy(
    bars: pd.DataFrame,
    window: int = ENTROPY_WINDOW,
    word_len: int = 5,
    n_quantiles: int = ENTROPY_ENCODE_QUANTILES,
) -> pd.DataFrame:
    """
    Rolling plug-in entropy on quantile-encoded returns.
    Uses quantile encoding with n_quantiles letters.
    """
    log_ret = np.diff(np.log(bars["close"].values), prepend=np.nan)
    n = len(log_ret)

    pi_vals = np.full(n, np.nan)

    print(f"  Computing plug-in entropy (window={window}, letters={n_quantiles})...")

    for t in range(window, n):
        rets = log_ret[t - window : t]
        valid = rets[~np.isnan(rets)]
        if len(valid) < word_len + 2:
            continue

        # Quantile encoding
        try:
            quantiles = np.percentile(valid, np.linspace(0, 100, n_quantiles + 1))
            codes = np.digitize(valid, quantiles[1:-1])  # values in [0, n_quantiles-1]
            msg = "".join(str(c) for c in codes)
            pi_vals[t] = _plug_in_entropy(msg, word_len)
        except Exception:
            continue

    bars["plugin_entropy"] = pi_vals
    return bars


# ============================================================================
# PART 4: MAIN PIPELINE
# ============================================================================

def compute_all_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature computations to volume bars DataFrame."""

    print(f"\n{'='*60}")
    print("COMPUTING FEATURES")
    print(f"{'='*60}\n")

    # 1. Order flow & VPIN
    print("  [1/11] Order Flow Imbalance & VPIN...")
    bars = compute_oi_vpin(bars)

    # 2. OI autocorrelation
    print("  [2/11] OI Serial Correlation...")
    bars = compute_oi_autocorr(bars)

    # 3. Kyle's Lambda
    print("  [3/11] Kyle's Lambda...")
    bars = compute_kyle_lambda(bars)

    # 4. Amihud's Lambda
    print("  [4/11] Amihud's Lambda...")
    bars = compute_amihud_lambda(bars)

    # 5. Hasbrouck's Lambda
    print("  [5/11] Hasbrouck's Lambda...")
    bars = compute_hasbrouck_lambda(bars)

    # 6. Roll Spread
    print("  [6/11] Roll Spread...")
    bars = compute_roll_spread(bars)

    # 7. Parkinson Volatility
    print("  [7/11] Parkinson Volatility...")
    bars = compute_parkinson_vol(bars)

    # 8. Corwin-Schultz
    print("  [8/11] Corwin-Schultz Spread...")
    bars = compute_corwin_schultz(bars)

    # 9. CSW CUSUM
    print("  [9/11] Chu-Stinchcombe-White CUSUM...")
    bars = compute_csw_cusum(bars)

    # 10. SADF  (computationally expensive — use step > 1 for speed)
    n_bars = len(bars)
    sadf_step = max(1, n_bars // 5000)  # aim for ~5000 SADF computations
    print(f"  [10/11] SADF (step={sadf_step})...")
    bars = compute_sadf(bars, step=sadf_step)

    # 11. Lempel-Ziv Entropy
    print("  [11/11] Lempel-Ziv Entropy & Plug-in Entropy...")
    bars = compute_lz_entropy(bars)
    bars = compute_plugin_entropy(bars)

    return bars


def main():
    t_start = _time.time()

    data_dir = DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Place BTCUSDT-aggTrades-*.csv/.zip files in ./data/")
        sys.exit(1)

    files = list_aggtrades_files(data_dir)
    if not files:
        print(f"ERROR: No aggTrades files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(files)} aggTrades files in {data_dir}")

    # --- Determine volume threshold ---
    threshold = VOLUME_THRESHOLD
    if threshold is None:
        print("\nAuto-computing volume threshold (1/50 of avg daily volume)...")
        avg_daily = estimate_daily_volume(data_dir)
        threshold = avg_daily / 50.0
        print(f"  Estimated avg daily volume: {avg_daily:,.2f} BTC")
        print(f"  Threshold (1/50): {threshold:,.2f} BTC")

    # --- Build volume bars ---
    bars = build_volume_bars(data_dir, threshold)

    if len(bars) < 50:
        print(f"WARNING: Only {len(bars)} bars. Consider lowering threshold.")

    # --- Compute features ---
    bars = compute_all_features(bars)

    # --- Save ---
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    bars.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"DONE in {_time.time() - t_start:.1f}s")
    print(f"Output: {output_path}")
    print(f"Shape:  {bars.shape}")
    print(f"Columns: {list(bars.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
