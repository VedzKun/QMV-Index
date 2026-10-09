"""
train.py — Model training orchestration.

Trains three model tiers for each prediction horizon:
  1. Ridge (linear baseline)
  2. GAM   (interpretable non-linear baseline)
  3. LightGBM (primary model)

Public API
----------
    get_feature_cols(df) -> list[str]
    train_ridge(X_tr, y_tr, cfg) -> Pipeline
    train_gam(X_tr, y_tr, cfg)   -> LinearGAM
    train_lgbm(X_tr, y_tr, X_val, y_val, cfg) -> lgb.Booster
    train_all(train, val, cfg, horizon_h) -> dict
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import lightgbm as lgb
from pygam import LinearGAM, s
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Columns that are NEVER features (IDs, raw text, target, timestamps)
_NON_FEATURE_COLS = {
    "story_id", "title", "url", "author", "created_at",
    "story_text", "target", "domain",
    "is_suspicious",  # flag, not a predictive feature
}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the sorted list of usable feature column names."""
    return sorted(c for c in df.columns if c not in _NON_FEATURE_COLS)


def train_ridge(X_tr: np.ndarray, y_tr: np.ndarray, cfg: dict) -> Pipeline:
    """Fit a StandardScaler + Ridge pipeline."""
    rcfg = cfg["models"]["ridge"]
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=rcfg["alpha"], max_iter=rcfg["max_iter"],
                         random_state=cfg["project"]["random_seed"])),
    ])
    pipe.fit(X_tr, y_tr)
    logger.info("Ridge fitted.")
    return pipe


def train_gam(X_tr: np.ndarray, y_tr: np.ndarray, cfg: dict):
    """Fit a LinearGAM with spline terms on each feature."""
    gcfg = cfg["models"]["gam"]
    n_features = X_tr.shape[1]
    terms = s(0)
    for i in range(1, n_features):
        terms = terms + s(i)
    gam = LinearGAM(terms, n_splines=gcfg["n_splines"], lam=gcfg["lam"])
    gam.fit(X_tr, y_tr)
    logger.info("GAM fitted.")
    return gam


def train_lgbm(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    cfg: dict,
    feature_names: list[str],
) -> lgb.Booster:
    """Train LightGBM with early stopping on validation set."""
    lcfg = dict(cfg["models"]["lgbm"])
    n_est = lcfg.pop("n_estimators")
    es    = lcfg.pop("early_stopping_rounds")

    dtrain = lgb.Dataset(X_tr,  label=y_tr,  feature_name=feature_names)
    dval   = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    callbacks = [
        lgb.early_stopping(es, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    booster = lgb.train(
        lcfg,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    logger.info("LightGBM fitted — best iteration: %d", booster.best_iteration)
    return booster


def train_all(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cfg: dict,
    horizon_h: int,
) -> dict:
    """
    Train all three model tiers for a given horizon.

    Returns
    -------
    dict with keys: 'ridge', 'gam', 'lgbm', 'feature_cols'
    """
    feat_cols = get_feature_cols(train)

    X_tr  = train[feat_cols].values.astype(float)
    y_tr  = train["target"].values
    X_val = val[feat_cols].values.astype(float)
    y_val = val["target"].values

    logger.info("Training T=%dh models on %d features, %d train / %d val samples",
                horizon_h, len(feat_cols), len(X_tr), len(X_val))

    ridge = train_ridge(X_tr, y_tr, cfg)
    gam   = train_gam(X_tr, y_tr, cfg)
    booster = train_lgbm(X_tr, y_tr, X_val, y_val, cfg, feat_cols)

    return {
        "ridge":        ridge,
        "gam":          gam,
        "lgbm":         booster,
        "feature_cols": feat_cols,
        "horizon_h":    horizon_h,
    }
