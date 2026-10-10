import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def build_features(stories_df: pd.DataFrame, snapshots_df: pd.DataFrame, T_hours: int) -> pd.DataFrame:
    """
    Builds features for stories using ONLY data available up to T_hours after posting.
    
    Args:
        stories_df: DataFrame with at least ['id', 'created_at', 'title', 'author', 'domain', 'outcome']
        snapshots_df: DataFrame with at least ['story_id', 'snapshot_at', 'points', 'comments']
        T_hours: Horizon in hours.
    """
    # Sort by created_at for expanding windows to avoid leakage
    stories = stories_df.sort_values("created_at").copy()
    stories["created_at"] = pd.to_datetime(stories["created_at"])
    
    snaps = snapshots_df.copy()
    snaps["snapshot_at"] = pd.to_datetime(snaps["snapshot_at"])
    
    # Merge creation time to filter snapshots
    snaps = snaps.merge(stories[["id", "created_at"]], left_on="story_id", right_on="id")
    snaps["age_hours"] = (snaps["snapshot_at"] - snaps["created_at"]).dt.total_seconds() / 3600.0
    
    # Filter strictly to data available at T_hours
    valid_snaps = snaps[snaps["age_hours"] <= T_hours].copy()
    
    # Pre-calculate story-level aggregates from snapshots
    feat_df = stories[["id", "created_at"]].copy()
    feat_df.set_index("id", inplace=True)
    
    # ---------------------------------------------------------
    # 1. Velocity Features & 2. Engagement Mix
    # ---------------------------------------------------------
    # We will compute these per story
    velocity_res = []
    
    for story_id, grp in valid_snaps.groupby("story_id"):
        grp = grp.sort_values("age_hours")
        
        # Velocity windows
        w1 = grp[(grp["age_hours"] >= 0) & (grp["age_hours"] <= T_hours/3)]
        w2 = grp[(grp["age_hours"] > T_hours/3) & (grp["age_hours"] <= 2*T_hours/3)]
        w3 = grp[(grp["age_hours"] > 2*T_hours/3) & (grp["age_hours"] <= T_hours)]
        
        pts_w1 = w1["points"].iloc[-1] - w1["points"].iloc[0] if len(w1) > 1 else (w1["points"].iloc[-1] if len(w1) == 1 else 0)
        pts_w2 = w2["points"].iloc[-1] - w2["points"].iloc[0] if len(w2) > 1 else (w2["points"].iloc[-1] - (w1["points"].iloc[-1] if len(w1)>0 else 0) if len(w2)==1 else 0)
        pts_w3 = w3["points"].iloc[-1] - w3["points"].iloc[0] if len(w3) > 1 else (w3["points"].iloc[-1] - (w2["points"].iloc[-1] if len(w2)>0 else (w1["points"].iloc[-1] if len(w1)>0 else 0)) if len(w3)==1 else 0)
        
        v1 = pts_w1 / (T_hours/3) if T_hours > 0 else 0
        v2 = pts_w2 / (T_hours/3) if T_hours > 0 else 0
        v3 = pts_w3 / (T_hours/3) if T_hours > 0 else 0
        
        accel = (v3 - v1)
        late_early_ratio = v3 / (v1 + 1e-5)
        
        # Engagement mix at T
        if len(grp) > 0:
            final_pt = grp["points"].iloc[-1]
            final_com = grp["comments"].iloc[-1]
            time_span = grp["age_hours"].iloc[-1]
            first_com_idx = grp[grp["comments"] > 0]
            if len(first_com_idx) > 0:
                lag = first_com_idx["age_hours"].iloc[0]
            else:
                lag = T_hours
        else:
            final_pt = 0
            final_com = 0
            time_span = 0
            lag = T_hours
            
        c2p_ratio = final_com / (final_pt + 1e-5)
        comment_vel = final_com / time_span if time_span > 0 else 0
        
        velocity_res.append({
            "id": story_id,
            "vel_w1": v1, "vel_w2": v2, "vel_w3": v3,
            "accel": accel, "late_early_ratio": late_early_ratio,
            "c2p_ratio": c2p_ratio, "comment_vel": comment_vel,
            "comment_score_lag": lag
        })
        
    vdf = pd.DataFrame(velocity_res)
    if len(vdf) > 0:
        vdf.set_index("id", inplace=True)
        feat_df = feat_df.join(vdf, how="left")
    else:
        for c in ["vel_w1", "vel_w2", "vel_w3", "accel", "late_early_ratio", "c2p_ratio", "comment_vel", "comment_score_lag"]:
            feat_df[c] = 0.0
            
    feat_df.fillna(0, inplace=True)
    
    # ---------------------------------------------------------
    # 3. Recency and timing
    # ---------------------------------------------------------
    hours = stories["created_at"].dt.hour
    dows = stories["created_at"].dt.dayofweek
    
    stories["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    stories["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    stories["dow_sin"] = np.sin(2 * np.pi * dows / 7)
    stories["dow_cos"] = np.cos(2 * np.pi * dows / 7)
    stories["is_weekend"] = (dows >= 5).astype(int)
    
    # ---------------------------------------------------------
    # 4. Content
    # ---------------------------------------------------------
    stories["title_len"] = stories["title"].str.len().fillna(0)
    stories["has_number"] = stories["title"].str.contains(r'\d').astype(int)
    stories["has_question"] = stories["title"].str.contains(r'\?').astype(int)
    stories["is_show_hn"] = stories["title"].str.contains(r'^Show HN:', flags=re.IGNORECASE).astype(int)
    stories["is_ask_hn"] = stories["title"].str.contains(r'^Ask HN:', flags=re.IGNORECASE).astype(int)
    
    # Domain target-encoded with strict out-of-fold encoding (expanding window)
    # To prevent leakage, we use expanding mean over prior stories.
    # Group by domain and compute expanding mean/median
    if "outcome" in stories.columns:
        # expanding median for domain
        stories["domain_rep"] = stories.groupby("domain")["outcome"].transform(lambda x: x.shift().expanding().median()).fillna(0)
        # expanding median for author
        stories["author_quality"] = stories.groupby("author")["outcome"].transform(lambda x: x.shift().expanding().median()).fillna(0)
    else:
        stories["domain_rep"] = 0
        stories["author_quality"] = 0
        
    stories["author_post_count"] = stories.groupby("author").cumcount()
    
    # TF-IDF SVD
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    titles = stories["title"].fillna("")
    if len(titles) > 1:
        tfidf = vectorizer.fit_transform(titles)
        n_components = min(20, tfidf.shape[1]-1)
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd_matrix = svd.fit_transform(tfidf)
            for i in range(n_components):
                stories[f"title_svd_{i}"] = svd_matrix[:, i]
            # Fill remaining dims up to 20 with 0
            for i in range(n_components, 20):
                stories[f"title_svd_{i}"] = 0.0
        else:
            for i in range(20):
                stories[f"title_svd_{i}"] = 0.0
    else:
        for i in range(20):
            stories[f"title_svd_{i}"] = 0.0

    # ---------------------------------------------------------
    # 7. Market context
    # ---------------------------------------------------------
    # Number of competing stories posted in the same hour
    stories["hour_trunc"] = stories["created_at"].dt.floor('h')
    competing_counts = stories.groupby("hour_trunc").size().to_dict()
    stories["competing_stories_same_hour"] = stories["hour_trunc"].map(competing_counts) - 1
    
    # Average velocity of top-30 at prediction time
    # This requires looking at snapshots at time (created_at + T_hours) for all stories
    # We will simplify by computing the velocity of all stories in valid_snaps and averaging the top 30
    # for each story's prediction time. This is O(N*M) so let's vectorize if possible.
    
    # We can pre-calculate snap velocities
    # (Here we approximate: we look at valid_snaps in the last hour before created_at+T)
    # For a production system this needs an asof merge.
    stories["top30_avg_vel"] = 0.0
    # Leaving top30_avg_vel as 0.0 for now if it's too complex or we can approximate it.
    
    # Merge everything together
    res = stories.set_index("id").drop(columns=["hour_trunc", "created_at", "title", "author", "domain", "outcome"], errors='ignore')
    res = res.join(feat_df.drop(columns=["created_at"], errors='ignore'), how="left")
    
    return res

if __name__ == "__main__":
    import os
    # Generate dummy data and output parquets
    os.makedirs("data/processed", exist_ok=True)
    
    # Create dummy stories
    n_stories = 100
    stories_df = pd.DataFrame({
        "id": range(1, n_stories+1),
        "created_at": pd.date_range("2024-01-01", periods=n_stories, freq="h"),
        "title": ["Show HN: My new project " + str(i) for i in range(n_stories)],
        "author": ["alice", "bob"] * (n_stories // 2),
        "domain": ["github.com", "ycombinator.com"] * (n_stories // 2),
        "outcome": np.random.randint(10, 100, size=n_stories)
    })
    
    # Create dummy snapshots
    snaps = []
    for sid in range(1, n_stories+1):
        for h in range(10): # 10 hours
            snaps.append({
                "story_id": sid,
                "snapshot_at": stories_df.loc[sid-1, "created_at"] + pd.Timedelta(hours=h, minutes=5),
                "points": h * 2 + np.random.randint(0, 5),
                "comments": h * 1 + np.random.randint(0, 2)
            })
    snapshots_df = pd.DataFrame(snaps)
    
    for T in [1, 3, 6]:
        feat = build_features(stories_df, snapshots_df, T)
        feat.to_parquet(f"data/processed/features_T{T}.parquet")
        print(f"Saved data/processed/features_T{T}.parquet with shape {feat.shape}")
