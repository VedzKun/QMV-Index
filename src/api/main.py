"""
main.py — QMV Index FastAPI application.
Run: uvicorn src.api.main:app --reload --port 8000
"""
import logging, json, time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import percentileofscore

from src.api.schemas import ValuateRequest, ValuateResponse, Driver
from src.api.model_store import load_model
from src.features.build_features import build_features

# ── structured logging ──────────────────────────────────────────────────
logging.basicConfig(
    format=json.dumps({
        "time": "%(asctime)s", "level": "%(levelname)s",
        "name": "%(name)s",    "msg":   "%(message)s",
    }),
    level=logging.INFO,
)
logger = logging.getLogger("qmv.api")

# ── lifespan: pre-warm all three models ────────────────────────────────
_models: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    for T in [1, 3, 6]:
        try:
            _models[T] = load_model(T)
            logger.info("Loaded model T=%s version=%s", T, _models[T]["version"])
        except Exception as exc:
            logger.warning("Could not load model T=%s: %s", T, exc)
    yield

app = FastAPI(
    title="QMV Index API",
    description="Quantitative Market Value scoring for Hacker News content.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── middleware: request timing ──────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    logger.info(
        "method=%s path=%s status=%s latency_ms=%.1f",
        request.method, request.url.path, response.status_code,
        (time.perf_counter() - t0) * 1000,
    )
    return response

# ── helpers ─────────────────────────────────────────────────────────────
def _build_feature_row(req: ValuateRequest, bundle: dict) -> np.ndarray:
    fc = bundle["feature_cols"]

    if req.snapshots:
        snaps_df = pd.DataFrame([s.model_dump() for s in req.snapshots])
        snaps_df["story_id"] = req.story_id or 0
        story_row = {
            "id": req.story_id or 0,
            "title": req.title or "",
            "url": req.url or "",
            "by": req.author or "unknown",
            "time": req.posted_at or int(time.time()),
            "type": "story",
        }
        stories_df = pd.DataFrame([story_row])
        feat_df = build_features(stories_df, snaps_df, req.T)
        row = feat_df.iloc[0]
        return np.ascontiguousarray(
            [[row[c] if c in row.index else 0.0 for c in fc]], dtype=np.float64
        )

    raise HTTPException(400, "story_id lookup not yet implemented; pass snapshots[].")


def _top_drivers(bundle: dict, X_row: np.ndarray) -> list[Driver]:
    m_ridge = bundle["m_ridge"]
    fc      = bundle["feature_cols"]
    coef    = m_ridge.named_steps["r"].coef_
    sc      = m_ridge.named_steps["sc"]
    z = (X_row[0] - sc.mean_) / (sc.scale_ + 1e-8)
    contribs = {feat: float(coef[i] * z[i]) for i, feat in enumerate(fc)}
    top5 = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    return [
        Driver(
            feature=f,
            contribution=round(c, 3),
            direction="positive" if c >= 0 else "negative",
        )
        for f, c in top5
    ]

# ── endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(_models.keys())}


@app.post("/valuate", response_model=ValuateResponse)
def valuate(req: ValuateRequest):
    T = req.T
    if T not in _models:
        raise HTTPException(503, f"Model for T={T} not available.")

    bundle = _models[T]
    X_row  = _build_feature_row(req, bundle)

    pred   = float(bundle["m_tree"].predict(X_row)[0])
    std    = bundle["resid_std"]
    qmv    = float(percentileofscore(bundle["all_preds"], pred, kind="weak"))
    drivers = _top_drivers(bundle, X_row)

    logger.info("valuate story_id=%s T=%s pred=%.3f qmv=%.1f", req.story_id, T, pred, qmv)

    return ValuateResponse(
        story_id        = req.story_id,
        T               = T,
        predicted_value = round(pred, 4),
        interval_low    = round(pred - 1.96 * std, 4),
        interval_high   = round(pred + 1.96 * std, 4),
        qmv_index       = round(qmv, 1),
        top_drivers     = drivers,
        model_version   = bundle["version"],
    )
