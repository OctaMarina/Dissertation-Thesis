"""
Reproduction pipeline (as close as possible) for:
"Deep learning and technical analysis in cryptocurrency market"
Finance Research Letters (2023)

Refactored for reproducible scientific testing:
- All parameters loaded from a JSON configuration file
- Dataset path and column mapping are configurable
- Extra features from the dataset are automatically included
- All trained models, scalers, and metadata are saved for later reuse
- Designed to be loaded from a Jupyter notebook for further analysis

Usage:
    python pipeline.py                          # uses config_default.json
    python pipeline.py --config my_config.json  # uses custom config
"""

from __future__ import annotations

import os
import sys
import math
import json
import copy
import random
import shutil
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, List, Optional, Any

import pickle

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# Optional tree models
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# Deep learning
try:
    import tensorflow as tf
    from tensorflow.keras import layers, callbacks, models, optimizers
    HAS_TF = True
except Exception:
    HAS_TF = False


# ============================================================
# Configuration — loaded from JSON
# ============================================================
@dataclass
class Config:
    """All experiment parameters. Built from a JSON config file."""

    experiment_name: str = "experiment_001"

    data_path: str = "btcusd_1-min_data.csv"

    # Column mapping: JSON key -> column name in your CSV
    # Required keys: timestamp, open, high, low, close, volume
    columns: Dict[str, str] = field(default_factory=lambda: {
        "timestamp": "Timestamp",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    # Any additional columns in the CSV to include as raw features.
    # These are included as-is (after numeric coercion) alongside the
    # standard feature sets. Use column names AS THEY APPEAR in the CSV.
    extra_features: List[str] = field(default_factory=list)

    start_date: str = "2017-12-01"
    end_date: str = "2022-08-31"

    lookback: int = 24
    horizon: int = 1

    train_ratio: float = 0.50
    val_ratio: float = 0.25
    test_ratio: float = 0.25

    alpha_grid_points: int = 1001
    min_signals: int = 50

    recall_range_long: Optional[Tuple[float, float]] = (0.08, 0.12)
    recall_range_short: Optional[Tuple[float, float]] = None

    sma_windows: Tuple[int, ...] = (15, 20, 25, 30)
    rsi_windows: Tuple[int, ...] = (15, 20, 25, 30)
    wr_window: int = 14
    so_window: int = 14
    mfi_window: int = 14

    enable_deep_models: bool = True
    enable_lgbm: bool = True
    seed: int = 42

    epochs: int = 500
    batch_size: int = 1024
    early_stop_patience: int = 25

    # Which models / strategies / input types / feature types to run
    model_list: List[str] = field(default_factory=lambda: [
        "logistic", "xgboost", "lgbm", "mlp", "gru", "lstm", "cnn"
    ])
    strategies: List[str] = field(default_factory=lambda: ["long", "short"])
    input_types: List[str] = field(default_factory=lambda: ["ohlc", "candle"])
    feature_types: List[str] = field(default_factory=lambda: ["raw", "extended"])

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert tuples back to lists for JSON serialisability
        for k in ("sma_windows", "rsi_windows", "recall_range_long", "recall_range_short"):
            v = d.get(k)
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def _coerce(val, target_type):
    """Helper to coerce JSON values into the types expected by Config."""
    if val is None:
        return None
    if target_type in (tuple, Tuple):
        return tuple(val) if val is not None else None
    return val


def load_config(path: str) -> Config:
    """Load a Config from a JSON file. Missing keys use defaults."""
    with open(path, "r") as f:
        raw = json.load(f)

    cfg = Config()

    simple_keys = [
        "experiment_name", "data_path", "start_date", "end_date",
        "lookback", "horizon", "train_ratio", "val_ratio", "test_ratio",
        "alpha_grid_points", "min_signals", "wr_window", "so_window",
        "mfi_window", "enable_deep_models", "enable_lgbm", "seed",
        "epochs", "batch_size", "early_stop_patience",
    ]
    for k in simple_keys:
        if k in raw:
            setattr(cfg, k, raw[k])

    if "columns" in raw:
        cfg.columns = raw["columns"]

    if "extra_features" in raw:
        cfg.extra_features = list(raw["extra_features"])

    for k in ("sma_windows", "rsi_windows"):
        if k in raw:
            setattr(cfg, k, tuple(raw[k]))

    for k in ("recall_range_long", "recall_range_short"):
        if k in raw:
            v = raw[k]
            setattr(cfg, k, tuple(v) if v is not None else None)

    if "models" in raw:
        cfg.model_list = list(raw["models"])
    if "strategies" in raw:
        cfg.strategies = list(raw["strategies"])
    if "input_types" in raw:
        cfg.input_types = list(raw["input_types"])
    if "feature_types" in raw:
        cfg.feature_types = list(raw["feature_types"])

    return cfg


# ============================================================
# Experiment directory structure
# ============================================================
class ExperimentDir:
    """Manages the output directory tree for one experiment run."""

    def __init__(self, base_dir: str, experiment_name: str):
        self.root = os.path.join(base_dir, experiment_name)
        self.models_dir = os.path.join(self.root, "models")
        self.preprocessing_dir = os.path.join(self.root, "preprocessing")
        self.results_dir = os.path.join(self.root, "results")
        self.checkpoints_dir = os.path.join(self.root, "checkpoints")

        for d in (self.root, self.models_dir, self.preprocessing_dir,
                  self.results_dir, self.checkpoints_dir):
            os.makedirs(d, exist_ok=True)

    def model_path(self, tag: str, ext: str) -> str:
        return os.path.join(self.models_dir, f"{tag}{ext}")

    def preprocessing_path(self, tag: str, ext: str) -> str:
        return os.path.join(self.preprocessing_dir, f"{tag}{ext}")

    def result_path(self, name: str) -> str:
        return os.path.join(self.results_dir, name)

    def checkpoint_path(self, tag: str) -> str:
        return os.path.join(self.checkpoints_dir, f"{tag}.keras")

    def save_config(self, cfg: Config, config_source_path: Optional[str] = None):
        """Save a copy of the resolved config into the experiment dir."""
        with open(os.path.join(self.root, "config_resolved.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        if config_source_path and os.path.isfile(config_source_path):
            shutil.copy2(config_source_path, os.path.join(self.root, "config_original.json"))


# ============================================================
# Reproducibility controls
# ============================================================
def set_global_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if HAS_TF:
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


# ============================================================
# Data loading / resampling
# ============================================================
def load_minute_ohlcv_csv(cfg: Config) -> pd.DataFrame:
    """
    Load a minute-resolution OHLCV CSV. Column names are mapped via
    cfg.columns. Any extra columns listed in cfg.extra_features are
    preserved and carried through the pipeline.
    """
    df = pd.read_csv(cfg.data_path)
    df.columns = [c.strip() for c in df.columns]

    # Build reverse mapping: csv_col_name -> canonical_name
    col_map = {}
    for canonical, csv_name in cfg.columns.items():
        # case-insensitive match
        matched = [c for c in df.columns if c.lower() == csv_name.lower()]
        if not matched:
            raise ValueError(
                f"Column '{csv_name}' (mapped to '{canonical}') not found in CSV. "
                f"Available: {df.columns.tolist()}"
            )
        col_map[matched[0]] = canonical

    # Rename mapped columns to canonical names
    df = df.rename(columns=col_map)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"After mapping, CSV still missing canonical columns: {sorted(missing)}")

    # Validate extra features exist
    for ef in cfg.extra_features:
        matches = [c for c in df.columns if c.lower() == ef.lower()]
        if not matches:
            raise ValueError(
                f"Extra feature '{ef}' not found in CSV. Available: {df.columns.tolist()}"
            )
        # Normalise the name if casing differs
        if matches[0] != ef:
            df = df.rename(columns={matches[0]: ef})

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Numeric safety for OHLCV + extra features
    numeric_cols = ["open", "high", "low", "close", "volume"] + list(cfg.extra_features)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.set_index("timestamp")
    return df


def resample_to_hourly(df_minute: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    From minute OHLCV -> hourly OHLCV, with proxy 'trades' count.
    Extra features are aggregated via mean.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    # Extra features: default to mean aggregation
    for ef in cfg.extra_features:
        if ef in df_minute.columns:
            agg[ef] = "mean"

    hourly = df_minute.resample("1h", label="left", closed="left").agg(agg)

    # proxy trades
    trades_proxy = (
        (df_minute["volume"] > 0).astype(int)
        .resample("1h", label="left", closed="left")
        .sum()
    )
    hourly["trades"] = trades_proxy

    hourly = hourly.dropna(subset=["open", "high", "low", "close"])
    hourly["volume"] = hourly["volume"].fillna(0.0)
    hourly["trades"] = hourly["trades"].fillna(0).astype(int)

    return hourly


def add_time_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index

    hour = idx.hour.astype(float)
    dow = idx.dayofweek.astype(float)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return out


# ============================================================
# Feature engineering
# ============================================================
def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    o = out["open"].values
    h = out["high"].values
    l = out["low"].values
    c = out["close"].values

    up = np.zeros_like(c, dtype=float)
    lo = np.zeros_like(c, dtype=float)
    body = np.abs(c - o)

    bullish = (c - o) > 0
    bearish = (c - o) < 0

    up[bullish] = h[bullish] - c[bullish]
    up[bearish] = h[bearish] - o[bearish]
    flat = ~(bullish | bearish)
    up[flat] = h[flat] - c[flat]

    lo[bullish] = o[bullish] - l[bullish]
    lo[bearish] = c[bearish] - l[bearish]
    lo[flat] = c[flat] - l[flat]

    out["candle_body"] = body
    out["upper_shadow"] = up
    out["lower_shadow"] = lo

    return out


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi_wilder(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(lower=0.0, upper=100.0)


def macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(close, fast) - ema(close, slow)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    hh = high.rolling(window=window, min_periods=window).max()
    ll = low.rolling(window=window, min_periods=window).min()
    denom = (hh - ll).replace(0.0, np.nan)
    wr = (hh - close) / denom * (-100.0)
    return wr


def stochastic_oscillator_so(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    hh = high.rolling(window=window, min_periods=window).max()
    ll = low.rolling(window=window, min_periods=window).min()
    denom = (hh - ll).replace(0.0, np.nan)
    k = (close - ll) / denom * 100.0
    d = k.rolling(window=3, min_periods=3).mean()
    so = k - d
    return so


def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume

    tp_prev = tp.shift(1)
    pos_mf = raw_mf.where(tp > tp_prev, 0.0)
    neg_mf = raw_mf.where(tp < tp_prev, 0.0)

    pos_sum = pos_mf.rolling(window=window, min_periods=window).sum()
    neg_sum = neg_mf.rolling(window=window, min_periods=window).sum().replace(0.0, np.nan)

    mfr = pos_sum / neg_sum
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    return mfi.clip(lower=0.0, upper=100.0)


def add_technical_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()

    for w in cfg.sma_windows:
        out[f"SMA_{w}"] = sma(out["close"], w)

    for w in cfg.rsi_windows:
        out[f"RSI_{w}"] = rsi_wilder(out["close"], w)

    out["MACD"] = macd_line(out["close"])
    out["WR"] = williams_r(out["high"], out["low"], out["close"], cfg.wr_window)
    out["SO"] = stochastic_oscillator_so(out["high"], out["low"], out["close"], cfg.so_window)
    out["MFI"] = money_flow_index(out["high"], out["low"], out["close"], out["volume"], cfg.mfi_window)

    return out


# ============================================================
# Labeling & dataset build
# ============================================================
def add_labels_and_returns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    fut_close = out["close"].shift(-cfg.horizon)
    ret = (fut_close / out["close"]) - 1.0
    out["future_return"] = ret

    out["y_long"] = (ret > 0.0).astype(int)
    out["y_short"] = (ret < 0.0).astype(int)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    input_type: str,
    feature_type: str,
    cfg: Config,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature matrix. Extra features from cfg.extra_features are always
    appended to the base feature set, regardless of input_type/feature_type.
    """
    if input_type not in {"ohlc", "candle"}:
        raise ValueError("input_type must be 'ohlc' or 'candle'")
    if feature_type not in {"raw", "extended"}:
        raise ValueError("feature_type must be 'raw' or 'extended'")

    base_cols = []
    if input_type == "ohlc":
        base_cols += ["open", "high", "low", "close"]
    else:
        base_cols += ["close", "candle_body", "upper_shadow", "lower_shadow"]

    base_cols += ["volume", "trades", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    ind_cols = []
    if feature_type == "extended":
        for w in [15, 20, 25, 30]:
            ind_cols.append(f"SMA_{w}")
        for w in [15, 20, 25, 30]:
            ind_cols.append(f"RSI_{w}")
        ind_cols += ["MACD", "WR", "SO", "MFI"]

    # Append any extra features from the dataset
    extra_cols = [ef for ef in cfg.extra_features if ef in df.columns]

    cols = base_cols + ind_cols + extra_cols
    Xdf = df[cols].copy()
    return Xdf, cols


def make_sequences(X: np.ndarray, y: np.ndarray, future_ret: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_obs, n_features)")
    n = X.shape[0]
    if n <= lookback:
        raise ValueError(f"Not enough observations ({n}) for lookback={lookback}")

    seqs = []
    ys = []
    rets = []
    for t in range(lookback - 1, n):
        seqs.append(X[t - lookback + 1 : t + 1, :])
        ys.append(y[t])
        rets.append(future_ret[t])

    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(rets, dtype=np.float32),
    )


def chronological_split(n_samples: int, cfg: Config) -> Tuple[slice, slice, slice]:
    n_train = int(n_samples * cfg.train_ratio)
    n_val = int(n_samples * cfg.val_ratio)
    n_test = n_samples - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Bad split sizes: train={n_train}, val={n_val}, test={n_test}")
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_samples)


def scale_sequences_fit_on_train(
    X_seq: np.ndarray,
    train_sl: slice,
    val_sl: slice,
    test_sl: slice,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    n_train = X_seq[train_sl].shape[0]
    lookback = X_seq.shape[1]
    n_feat = X_seq.shape[2]

    scaler = StandardScaler()
    train_2d = X_seq[train_sl].reshape(n_train * lookback, n_feat)
    scaler.fit(train_2d)

    def transform_block(block: np.ndarray) -> np.ndarray:
        b = block.reshape(block.shape[0] * lookback, n_feat)
        b = scaler.transform(b)
        return b.reshape(block.shape[0], lookback, n_feat).astype(np.float32)

    X_train = transform_block(X_seq[train_sl])
    X_val = transform_block(X_seq[val_sl])
    X_test = transform_block(X_seq[test_sl])
    return X_train, X_val, X_test, scaler


# ============================================================
# Thresholding & metrics
# ============================================================
def choose_alpha_max_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: Config,
    recall_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)

    alphas = np.linspace(0.0, 1.0, cfg.alpha_grid_points)

    best_alpha = 1.0
    best_prec = -1.0
    best_rec = -1.0

    lo, hi = (None, None)
    if recall_range is not None:
        lo, hi = recall_range

    for a in alphas:
        y_pred = (y_prob >= a).astype(int)
        n_sig = int(y_pred.sum())
        if n_sig < cfg.min_signals:
            continue

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if recall_range is not None and (rec < lo or rec > hi):
            continue

        if (prec > best_prec) or (math.isclose(prec, best_prec) and a > best_alpha):
            best_alpha = float(a)
            best_prec = float(prec)
            best_rec = float(rec)

    # Fallback
    if best_prec < 0.0:
        for a in alphas:
            y_pred = (y_prob >= a).astype(int)
            n_sig = int(y_pred.sum())
            if n_sig < cfg.min_signals:
                continue
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            if (prec > best_prec) or (math.isclose(prec, best_prec) and a > best_alpha):
                best_alpha = float(a)
                best_prec = float(prec)
                best_rec = float(rec)

    return best_alpha, best_prec, best_rec


def strategy_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    future_ret: np.ndarray,
    direction: int,
    alpha: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= alpha).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    sig = y_pred.astype(bool)
    n_sig = int(sig.sum())

    if n_sig == 0:
        return {
            "precision": float(prec),
            "recall": float(rec),
            "avg_return": 0.0,
            "sharpe": 0.0,
        }

    step_returns = (direction * future_ret[sig]).astype(float)
    total_factor = float(np.prod(1.0 + step_returns))

    avg_ret = (total_factor - 1.0) / n_sig

    std_ret = float(np.std(step_returns, ddof=1)) if n_sig > 1 else 0.0
    sharpe = (avg_ret / std_ret) if std_ret > 0 else 0.0

    return {
        "precision": float(prec),
        "recall": float(rec),
        "avg_return": float(avg_ret),
        "sharpe": float(sharpe),
    }


# ============================================================
# Models
# ============================================================
def flatten_3d_to_2d(X3: np.ndarray) -> np.ndarray:
    return X3.reshape(X3.shape[0], -1)


def build_mlp(input_dim: int) -> "tf.keras.Model":
    inp = layers.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")
    return model


def build_rnn(model_type: str, lookback: int, n_feat: int) -> "tf.keras.Model":
    inp = layers.Input(shape=(lookback, n_feat), dtype=tf.float32)
    if model_type == "lstm":
        x = layers.LSTM(8, activation="relu", dropout=0.5)(inp)
    elif model_type == "gru":
        x = layers.GRU(8, activation="relu", dropout=0.5)(inp)
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")
    return model


def build_cnn(lookback: int, n_feat: int) -> "tf.keras.Model":
    inp = layers.Input(shape=(lookback, n_feat), dtype=tf.float32)
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv1D(filters=8, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")
    return model


def train_keras_model(
    model: "tf.keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
    checkpoint_path: str,
) -> "tf.keras.Model":
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.early_stop_patience,
            restore_best_weights=True,
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=cbs,
    )
    return model


def predict_proba_model(
    model_name: str,
    model_obj,
    X: np.ndarray,
) -> np.ndarray:
    if model_name in {"logistic", "xgboost", "lgbm"}:
        proba = model_obj.predict_proba(X)[:, 1]
        return proba.astype(float)
    else:
        p = model_obj.predict(X, verbose=0).reshape(-1)
        return p.astype(float)


# ============================================================
# Artifact saving / loading helpers
# ============================================================
def _combo_tag(model_name: str, strategy: str, input_type: str, feature_type: str) -> str:
    return f"{model_name}_{strategy}_{input_type}_{feature_type}"


def save_model(model_name: str, model_obj, path_no_ext: str) -> str:
    """Save model to disk. Returns the actual path written."""
    if model_name in {"logistic", "xgboost", "lgbm"}:
        path = path_no_ext + ".pkl"
        with open(path, "wb") as f:
            pickle.dump(model_obj, f)
        return path
    else:
        path = path_no_ext + ".keras"
        model_obj.save(path)
        return path


def load_model_from_disk(model_name: str, path_no_ext: str):
    """Load a previously saved model."""
    if model_name in {"logistic", "xgboost", "lgbm"}:
        with open(path_no_ext + ".pkl", "rb") as f:
            return pickle.load(f)
    else:
        return tf.keras.models.load_model(path_no_ext + ".keras")


def save_preprocessing(
    scaler: StandardScaler,
    feature_cols: List[str],
    alpha: float,
    direction: int,
    lookback: int,
    model_name: str,
    tag: str,
    exp_dir: ExperimentDir,
) -> None:
    """Save scaler and metadata needed to reproduce inference."""
    base = exp_dir.preprocessing_path(tag, "")

    with open(base + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_columns": feature_cols,
        "alpha": alpha,
        "direction": direction,
        "lookback": lookback,
        "model_name": model_name,
        "tag": tag,
    }
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_preprocessing(tag: str, exp_dir: ExperimentDir) -> Tuple[StandardScaler, dict]:
    """Load scaler and metadata for a given combo tag."""
    base = exp_dir.preprocessing_path(tag, "")
    with open(base + "_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(base + "_meta.json", "r") as f:
        meta = json.load(f)
    return scaler, meta


# ============================================================
# Experiment runner
# ============================================================
def prepare_full_dataframe(cfg: Config) -> pd.DataFrame:
    df_min = load_minute_ohlcv_csv(cfg)
    df_hr = resample_to_hourly(df_min, cfg)

    # filter to period
    start = pd.to_datetime(cfg.start_date, utc=True)
    end = pd.to_datetime(cfg.end_date, utc=True) + pd.Timedelta(days=1)
    df_hr = df_hr.loc[(df_hr.index >= start) & (df_hr.index < end)].copy()

    df_hr = add_time_cyclical_features(df_hr)
    df_hr = add_candle_features(df_hr)
    df_hr = add_technical_indicators(df_hr, cfg)
    df_hr = add_labels_and_returns(df_hr, cfg)

    df_hr = df_hr.dropna()
    return df_hr


def fit_and_eval_one(
    df: pd.DataFrame,
    cfg: Config,
    model_name: str,
    input_type: str,
    feature_type: str,
    strategy: str,
    exp_dir: ExperimentDir,
) -> Dict[str, object]:
    """
    Fits one model for one combination, saves model + preprocessing,
    and returns paper-like metrics on val/test.
    """
    tag = _combo_tag(model_name, strategy, input_type, feature_type)

    Xdf, cols = build_feature_matrix(df, input_type=input_type, feature_type=feature_type, cfg=cfg)

    if strategy == "long":
        y = df["y_long"].values.astype(int)
        direction = +1
        recall_range = cfg.recall_range_long
    elif strategy == "short":
        y = df["y_short"].values.astype(int)
        direction = -1
        recall_range = cfg.recall_range_short
    else:
        raise ValueError("strategy must be 'long' or 'short'")

    future_ret = df["future_return"].values.astype(float)
    X = Xdf.values.astype(np.float32)

    X_seq, y_seq, r_seq = make_sequences(X, y, future_ret, cfg.lookback)
    train_sl, val_sl, test_sl = chronological_split(len(y_seq), cfg)
    X_train3, X_val3, X_test3, scaler = scale_sequences_fit_on_train(
        X_seq, train_sl, val_sl, test_sl
    )
    y_train, y_val, y_test = y_seq[train_sl], y_seq[val_sl], y_seq[test_sl]
    r_val, r_test = r_seq[val_sl], r_seq[test_sl]

    # Prepare input shapes
    if model_name in {"logistic", "xgboost", "lgbm", "mlp"}:
        X_train = flatten_3d_to_2d(X_train3)
        X_val = flatten_3d_to_2d(X_val3)
        X_test = flatten_3d_to_2d(X_test3)
    else:
        X_train, X_val, X_test = X_train3, X_val3, X_test3

    # Fit
    fitted = None

    if model_name == "logistic":
        clf = LogisticRegression(max_iter=2000, n_jobs=None)
        clf.fit(X_train, y_train)
        fitted = clf

    elif model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed")
        clf = xgb.XGBClassifier(
            n_estimators=100,
            random_state=cfg.seed,
            n_jobs=-1,
            max_depth=6,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        fitted = clf

    elif model_name == "lgbm":
        if not (HAS_LGBM and cfg.enable_lgbm):
            raise RuntimeError("lightgbm is not installed or disabled")
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=cfg.seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        fitted = clf

    else:
        if not HAS_TF:
            raise RuntimeError("tensorflow is not installed")

        ckpt_path = exp_dir.checkpoint_path(tag)

        if model_name == "mlp":
            net = build_mlp(input_dim=X_train.shape[1])
        elif model_name == "gru":
            net = build_rnn("gru", cfg.lookback, X_train.shape[2])
        elif model_name == "lstm":
            net = build_rnn("lstm", cfg.lookback, X_train.shape[2])
        elif model_name == "cnn":
            net = build_cnn(cfg.lookback, X_train.shape[2])
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        net = train_keras_model(net, X_train, y_train, X_val, y_val, cfg, ckpt_path)
        fitted = net

    # Predict on val/test
    p_val = predict_proba_model(model_name, fitted, X_val)
    p_test = predict_proba_model(model_name, fitted, X_test)

    # Choose alpha
    alpha, val_prec_at_alpha, val_rec_at_alpha = choose_alpha_max_precision(
        y_val, p_val, cfg, recall_range=recall_range
    )

    # Compute metrics
    val_metrics = strategy_metrics(y_val, p_val, r_val, direction=direction, alpha=alpha)
    test_metrics = strategy_metrics(y_test, p_test, r_test, direction=direction, alpha=alpha)

    # ---- SAVE ARTIFACTS ----
    model_path = save_model(
        model_name, fitted,
        exp_dir.model_path(tag, ""),
    )
    save_preprocessing(
        scaler=scaler,
        feature_cols=cols,
        alpha=alpha,
        direction=direction,
        lookback=cfg.lookback,
        model_name=model_name,
        tag=tag,
        exp_dir=exp_dir,
    )

    return {
        "Model": model_name.upper(),
        "InputType": input_type,
        "FeatureType": feature_type,
        "Strategy": strategy,
        "Alpha": alpha,
        "Val_Sharpe": val_metrics["sharpe"],
        "Val_Precision": val_metrics["precision"],
        "Val_Recall": val_metrics["recall"],
        "Val_AvgReturn": val_metrics["avg_return"],
        "Test_Sharpe": test_metrics["sharpe"],
        "Test_Precision": test_metrics["precision"],
        "Test_Recall": test_metrics["recall"],
        "Test_AvgReturn": test_metrics["avg_return"],
        "Model_Path": model_path,
    }


def run_full_study(cfg: Config, config_path: Optional[str] = None) -> pd.DataFrame:
    set_global_seeds(cfg.seed)

    exp_dir = ExperimentDir("experiments", cfg.experiment_name)
    exp_dir.save_config(cfg, config_source_path=config_path)

    df = prepare_full_dataframe(cfg)
    if len(df) < 1000:
        raise RuntimeError(
            f"Too few hourly rows after filtering: {len(df)}. "
            "Check start/end dates and data coverage."
        )

    # Save the processed hourly dataframe for notebook reuse
    df.to_csv(os.path.join(exp_dir.root, "hourly_data.parquet"))

    # Resolve which models are actually available
    active_models = []
    for m in cfg.model_list:
        if m == "xgboost" and not HAS_XGB:
            print(f"  [SKIP] xgboost not installed")
            continue
        if m == "lgbm" and (not HAS_LGBM or not cfg.enable_lgbm):
            print(f"  [SKIP] lightgbm not installed or disabled")
            continue
        if m in ("mlp", "gru", "lstm", "cnn"):
            if not cfg.enable_deep_models:
                print(f"  [SKIP] {m} (enable_deep_models=False)")
                continue
            if not HAS_TF:
                print(f"  [SKIP] {m} (tensorflow not installed)")
                continue
        active_models.append(m)

    combos = []
    for strategy in cfg.strategies:
        for input_type in cfg.input_types:
            for feature_type in cfg.feature_types:
                combos.append((strategy, input_type, feature_type))

    results = []
    total = len(active_models) * len(combos)
    done = 0

    for model_name in active_models:
        for (strategy, input_type, feature_type) in combos:
            done += 1
            print(f"[{done}/{total}] {model_name} | {strategy} | {input_type} | {feature_type}")

            try:
                row = fit_and_eval_one(
                    df=df,
                    cfg=cfg,
                    model_name=model_name,
                    input_type=input_type,
                    feature_type=feature_type,
                    strategy=strategy,
                    exp_dir=exp_dir,
                )
                results.append(row)
            except Exception as e:
                results.append({
                    "Model": model_name.upper(),
                    "InputType": input_type,
                    "FeatureType": feature_type,
                    "Strategy": strategy,
                    "Alpha": np.nan,
                    "Val_Sharpe": np.nan,
                    "Val_Precision": np.nan,
                    "Val_Recall": np.nan,
                    "Val_AvgReturn": np.nan,
                    "Test_Sharpe": np.nan,
                    "Test_Precision": np.nan,
                    "Test_Recall": np.nan,
                    "Test_AvgReturn": np.nan,
                    "Model_Path": "",
                    "Error": str(e),
                })
                print(f"  -> ERROR: {e}")

    res = pd.DataFrame(results)

    if "Test_Precision" in res.columns:
        res = res.sort_values(
            ["Strategy", "Test_Precision"], ascending=[True, False]
        ).reset_index(drop=True)

    # Save results
    res.to_csv(exp_dir.result_path("study_results_all.csv"), index=False)
    res[res["Strategy"] == "long"].to_csv(exp_dir.result_path("study_results_long.csv"), index=False)
    res[res["Strategy"] == "short"].to_csv(exp_dir.result_path("study_results_short.csv"), index=False)

    return res


def format_like_paper_table(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    view = df[df["Strategy"] == strategy].copy()
    cols = [
        "Model", "InputType", "FeatureType", "Strategy",
        "Val_Sharpe", "Val_Precision", "Val_Recall", "Val_AvgReturn",
        "Test_Sharpe", "Test_Precision", "Test_Recall", "Test_AvgReturn",
        "Alpha",
    ]
    cols = [c for c in cols if c in view.columns]
    view = view[cols].copy()

    for c in ["Val_Precision", "Val_Recall", "Test_Precision", "Test_Recall"]:
        if c in view.columns:
            view[c] = (view[c] * 100.0).round(2).astype(str) + "%"

    for c in ["Val_AvgReturn", "Test_AvgReturn"]:
        if c in view.columns:
            view[c] = (view[c] * 100.0).round(3).astype(str) + "%"

    for c in ["Val_Sharpe", "Test_Sharpe", "Alpha"]:
        if c in view.columns:
            view[c] = view[c].astype(float).round(2)

    return view


# ============================================================
# CLI entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run the deep learning + TA cryptocurrency study"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config_default.json",
        help="Path to JSON configuration file (default: config_default.json)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("=" * 60)
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Config file: {args.config}")
    print(f"Dataset: {cfg.data_path}")
    print(f"Extra features: {cfg.extra_features}")
    print("=" * 60)
    print("Resolved config:")
    print(json.dumps(cfg.to_dict(), indent=2))
    print("=" * 60)

    res = run_full_study(cfg, config_path=args.config)

    exp_root = os.path.join("experiments", cfg.experiment_name, "results")

    long_tbl = format_like_paper_table(res, "long")
    short_tbl = format_like_paper_table(res, "short")

    long_tbl.to_csv(os.path.join(exp_root, "table_like_paper_long.csv"), index=False)
    short_tbl.to_csv(os.path.join(exp_root, "table_like_paper_short.csv"), index=False)

    print(f"\nAll outputs saved under: experiments/{cfg.experiment_name}/")
    print("  models/          - all trained models (.pkl / .keras)")
    print("  preprocessing/   - scalers + metadata (.pkl / .json)")
    print("  results/         - CSV result tables")
    print("  config_resolved.json  - exact config used")
    print("  hourly_data.parquet   - processed hourly dataframe")


if __name__ == "__main__":
    main()
