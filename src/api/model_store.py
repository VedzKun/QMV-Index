"""
model_store.py — loads and caches the trained model artefacts.
"""
import pickle, hashlib, time, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger("qmv.model_store")

MODELS_DIR = Path(__file__).parents[2] / "models"
DATA_DIR   = Path(__file__).parents[2] / "data" / "processed"
DROP_COLS  = {"id", "title", "url", "by", "time", "type", "target", "domain"}

_cache: dict = {}


def _get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in DROP_COLS]


def _train_fresh(T: int = 3):
    """Fallback: train on the processed parquet when no saved model exists."""
    logger.warning("No saved model found for T=%s — training fresh.", T)
    df = pd.read_parquet(DATA_DIR / f"features_t{T}.parquet").sort_values("time")
    fc = _get_feature_cols(df)
    split = int(len(df) * 0.8)
    train = df.iloc[:split]

    Xtr = np.ascontiguousarray(train[fc].values, dtype=np.float64)
    ytr = np.ascontiguousarray(train["target"].values, dtype=np.float64)

    mc = [1 if c in {"v1", "v2", "v3", "acceleration",
                      "author_hist_median", "domain_hist_median"} else 0 for c in fc]
    m_tree = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.05, max_depth=5,
        monotonic_cst=mc, random_state=42)
    m_tree.fit(Xtr, ytr)

    m_ridge = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=10))])
    m_ridge.fit(Xtr, ytr)

    # Estimate residual std on hold-out for confidence intervals
    Xte = np.ascontiguousarray(df.iloc[split:][fc].values, dtype=np.float64)
    yte = df.iloc[split:]["target"].values
    resid_std = float(np.std(m_tree.predict(Xte) - yte))

    # All-data predictions for QMV percentile baseline
    all_preds = m_tree.predict(np.ascontiguousarray(df[fc].values, dtype=np.float64))

    return {
        "T": T,
        "feature_cols": fc,
        "m_tree": m_tree,
        "m_ridge": m_ridge,
        "resid_std": resid_std,
        "all_preds": all_preds,
        "version": hashlib.sha1(str(time.time()).encode()).hexdigest()[:8],
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def load_model(T: int = 3) -> dict:
    """Return (possibly cached) model bundle for horizon T."""
    if T in _cache:
        return _cache[T]

    pkl_path = MODELS_DIR / f"qmv_t{T}.pkl"
    if pkl_path.exists():
        logger.info("Loading model from %s", pkl_path)
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
    else:
        bundle = _train_fresh(T)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(bundle, f)

    _cache[T] = bundle
    return bundle
