"""Pydantic request / response schemas for the QMV API."""
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class SnapshotInput(BaseModel):
    hours_since_post: float = Field(..., ge=0, le=72)
    score: int              = Field(..., ge=0)
    descendants: int        = Field(..., ge=0)


class ValuateRequest(BaseModel):
    story_id:  Optional[int]            = None
    title:     Optional[str]            = None
    url:       Optional[str]            = None
    author:    Optional[str]            = None
    posted_at: Optional[int]            = None   # Unix timestamp
    snapshots: Optional[list[SnapshotInput]] = None
    T:         int = Field(3, ge=1, le=6)

    @model_validator(mode="after")
    def require_id_or_snapshots(self):
        if self.story_id is None and not self.snapshots:
            raise ValueError("Provide either story_id or snapshots[].")
        return self


class Driver(BaseModel):
    feature: str
    contribution: float
    direction: str   # "positive" | "negative"


class ValuateResponse(BaseModel):
    story_id:        Optional[int]
    T:               int
    predicted_value: float
    interval_low:    float
    interval_high:   float
    qmv_index:       float          # 0-100 percentile
    top_drivers:     list[Driver]
    model_version:   str
