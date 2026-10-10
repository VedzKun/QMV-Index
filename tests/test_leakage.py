import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add src to path if needed, though pytest usually handles it if run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.build_features import build_features

@pytest.fixture
def dummy_data():
    n_stories = 20
    stories_df = pd.DataFrame({
        "id": range(1, n_stories+1),
        "created_at": pd.date_range("2024-01-01", periods=n_stories, freq="h"),
        "title": ["Show HN: Test " + str(i) for i in range(n_stories)],
        "author": ["alice", "bob"] * (n_stories // 2),
        "domain": ["github.com", "yc.com"] * (n_stories // 2),
        "outcome": [10 * i for i in range(1, n_stories+1)]
    })
    
    snaps = []
    for sid in range(1, n_stories+1):
        for h in range(10): # up to 10 hours of snapshots
            snaps.append({
                "story_id": sid,
                "snapshot_at": stories_df.loc[sid-1, "created_at"] + pd.Timedelta(hours=h, minutes=5),
                "points": h * 5 + 1,
                "comments": h * 2
            })
    snapshots_df = pd.DataFrame(snaps)
    return stories_df, snapshots_df

def test_truncation_leakage(dummy_data):
    """
    Truncate snapshots to T hours and assert that features are identical 
    to those computed from the full history for T hours.
    """
    stories_df, snapshots_df = dummy_data
    T = 3
    
    # Run with full snapshots
    feat_full = build_features(stories_df, snapshots_df, T_hours=T)
    
    # Run with explicitly truncated snapshots
    # to simulate the database only containing data up to T hours for each story
    # Wait, the feature function ALREADY filters by age_hours <= T_hours.
    # To truly test leakage, we truncate the snapshots_df itself before passing it in.
    
    # We want to truncate snapshots to T hours *globally* for a specific story?
    # No, the requirement: "for random stories, truncate snapshots to T hours and assert that features are identical to those computed from the full history."
    # Let's filter the snapshots_df so that for each story, we only pass snapshots <= T hours
    
    snaps_trunc = []
    for _, snap in snapshots_df.iterrows():
        story = stories_df[stories_df["id"] == snap["story_id"]].iloc[0]
        age = (snap["snapshot_at"] - story["created_at"]).total_seconds() / 3600.0
        if age <= T:
            snaps_trunc.append(snap)
            
    snaps_trunc_df = pd.DataFrame(snaps_trunc)
    feat_trunc = build_features(stories_df, snaps_trunc_df, T_hours=T)
    
    pd.testing.assert_frame_equal(feat_full, feat_trunc, check_dtype=False)

def test_author_domain_leakage(dummy_data):
    """
    Assert that author/domain features never include the story's own outcome.
    """
    stories_df, snapshots_df = dummy_data
    T = 3
    
    feat = build_features(stories_df, snapshots_df, T_hours=T)
    
    # The first story by alice should have 0 author_quality (no prior stories)
    alice_stories = feat[stories_df.set_index("id")["author"] == "alice"]
    assert alice_stories.iloc[0]["author_quality"] == 0
    
    # The second story by alice should have author_quality == outcome of the first story
    # outcome of first story (id=1) is 10.
    assert alice_stories.iloc[1]["author_quality"] == 10
    
    # The first story for github.com should have 0 domain_rep
    github_stories = feat[stories_df.set_index("id")["domain"] == "github.com"]
    assert github_stories.iloc[0]["domain_rep"] == 0
    
    # The second story for github.com should have domain_rep == outcome of first story
    assert github_stories.iloc[1]["domain_rep"] == 10

