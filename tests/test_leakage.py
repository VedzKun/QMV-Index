"""
test_leakage.py
"""
import pandas as pd
import numpy as np
from src.features.build_features import build_features

def _mock_data():
    stories = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Story 1", "Story 2", "Story 3"],
        "url": ["https://a.com", "https://b.com", "https://a.com"],
        "by": ["alice", "bob", "alice"],
        # story 1 at T=1000. 72h snapshot happens at 1000 + 259200 = 260200
        # story 2 at T=2000.
        # story 3 at T=270200. T=3h prediction time happens at 270200 + 10800 = 281000
        # since 260200 <= 281000, story 1's 72h target is fully available when predicting story 3.
        "time": [1000, 2000, 270200],
        "type": ["story"]*3
    })
    
    # We create snapshots for 1h, 3h, 72h
    snaps = []
    for idx, sid in enumerate([1, 2, 3]):
        post_time = stories.iloc[idx]["time"]
        for h in [1.0, 3.0, 72.0]:
            snaps.append({
                "story_id": sid,
                "snapshot_time": post_time + h * 3600,
                "hours_since_post": h,
                "score": h * 10 * sid,
                "descendants": h * 2 * sid
            })
    
    snapshots = pd.DataFrame(snaps)
    return stories, snapshots


def test_time_truncation_leakage():
    """Assert features at T=3 are identical whether built from full DB or truncated DB."""
    stories, snapshots = _mock_data()
    
    # 1. Build features using full history (the function SHOULD truncate internally)
    feat_full = build_features(stories, snapshots, T_hours=3.0)
    
    # 2. To test leakage, we must test predicting a SPECIFIC story.
    # If we truncate the entire DB, we wipe out historical 72h targets.
    # So we truncate ONLY the future snapshots of the target story (id=3).
    snapshots_truncated = snapshots.copy()
    # Remove any snapshot for story 3 that is > 3h
    mask_to_remove = (snapshots_truncated["story_id"] == 3) & (snapshots_truncated["hours_since_post"] > 3.0)
    snapshots_truncated = snapshots_truncated[~mask_to_remove]
    
    feat_trunc = build_features(stories, snapshots_truncated, T_hours=3.0)
    
    # The outputs must be perfectly identical
    pd.testing.assert_frame_equal(feat_full, feat_trunc)


def test_author_reputation_leakage():
    """Assert author historical median only includes PRIOR stories."""
    stories, snapshots = _mock_data()
    
    feat = build_features(stories, snapshots, T_hours=3.0)
    feat = feat.set_index("id")
    
    # Alice's 1st story (id=1): No history
    assert feat.loc[1, "author_hist_median"] == 0.0
    
    # Bob's 1st story (id=2): No history
    assert feat.loc[2, "author_hist_median"] == 0.0
    
    # Alice's 2nd story (id=3): History should ONLY be the final value of id=1
    # id=1 at 72h: score=720, desc=144. Target = log1p(720 + 2*144) = log1p(1008)
    val_id1 = np.log1p(720 + 2*144)
    assert np.isclose(feat.loc[3, "author_hist_median"], val_id1)
