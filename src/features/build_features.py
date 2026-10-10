"""
build_features.py — Feature engineering for QMV Index.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import urllib.parse
from scipy.interpolate import interp1d
import argparse
from pathlib import Path
import sqlite3
from src.collect.db import DB_DEFAULT

def extract_domain(url):
    if not url or pd.isna(url) or url == "": return "self"
    try:
        netloc = urllib.parse.urlparse(url).netloc
        return netloc.replace("www.", "") if netloc else "self"
    except:
        return "self"

def interpolate_values(story_snaps, eval_times, col):
    x = story_snaps["hours_since_post"].values
    y = story_snaps[col].values
    
    if 0 not in x:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
        
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    
    if len(x) < 2:
        return np.zeros(len(eval_times))
        
    f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
    return f(eval_times)

def build_features(stories_df: pd.DataFrame, snapshots_df: pd.DataFrame, T_hours: float) -> pd.DataFrame:
    """Build features using ONLY data strictly available at T_hours after post time."""
    
    # 1. Truncate strict time window for the CURRENT stories being predicted
    # Wait, the rule is: any snapshot in the DB is available if snapshot_time <= story.time + T_hours
    # For velocity metrics of the current story, we only want its snapshots up to T_hours.
    snaps = snapshots_df[snapshots_df["hours_since_post"] <= T_hours + 1e-5].copy()
    
    df = stories_df.copy()
    df["predict_time"] = df["time"] + T_hours * 3600
    
    # 2. Velocity and Interpolation
    t_eval = np.array([T_hours/3, 2*T_hours/3, T_hours])
    
    def get_vels(group):
        pts = interpolate_values(group, t_eval, "score")
        comms = interpolate_values(group, t_eval, "descendants")
        return pd.Series({
            "pts_t1": pts[0], "pts_t2": pts[1], "pts_t3": pts[2],
            "comms_t3": comms[2]
        })
        
    if len(snaps) > 0:
        interp_df = snaps.groupby("story_id").apply(get_vels, include_groups=False).reset_index()
        df = df.merge(interp_df, left_on="id", right_on="story_id", how="left").fillna(0)
    else:
        for c in ["pts_t1", "pts_t2", "pts_t3", "comms_t3"]: df[c] = 0.0
        
    df["v1"] = df["pts_t1"] / (T_hours/3)
    df["v2"] = (df["pts_t2"] - df["pts_t1"]) / (T_hours/3)
    df["v3"] = (df["pts_t3"] - df["pts_t2"]) / (T_hours/3)
    
    df["acceleration"] = df["v3"] - df["v1"]
    df["late_early_vel_ratio"] = df["v3"] / (df["v1"] + 1e-5)
    
    # Engagement mix
    df["comments_points_ratio"] = df["comms_t3"] / (df["pts_t3"] + 1e-5)
    df["comment_velocity"] = df["comms_t3"] / T_hours
    df["comment_lag"] = df["pts_t3"] - df["comms_t3"] 
    
    # 3. Recency & Timing
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour"] = df["time_utc"].dt.hour
    df["dow"] = df["time_utc"].dt.dayofweek
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    
    # 4. Content
    df["title_len"] = df["title"].str.len().fillna(0)
    df["has_question"] = df["title"].str.contains(r'\?', na=False).astype(int)
    df["has_number"] = df["title"].str.contains(r'\d', na=False).astype(int)
    df["is_ask_hn"] = df["title"].str.startswith("Ask HN:").astype(int)
    df["is_show_hn"] = df["title"].str.startswith("Show HN:").astype(int)
    
    df["domain"] = df["url"].apply(extract_domain)
    
    # TF-IDF SVD
    titles = df["title"].fillna("")
    if len(titles) > 1:
        tfidf = TfidfVectorizer(max_features=1000, stop_words="english")
        X_tfidf = tfidf.fit_transform(titles)
        n_comp = min(20, X_tfidf.shape[1]-1)
        if n_comp > 0:
            svd = TruncatedSVD(n_components=n_comp, random_state=42)
            X_svd = svd.fit_transform(X_tfidf)
            for i in range(X_svd.shape[1]):
                df[f"title_svd_{i}"] = X_svd[:, i]
    
    # 5 & 6. Author Quality and Domain Reputation (STRICT LEAKAGE PREVENTION)
    snaps_72 = snapshots_df[np.isclose(snapshots_df["hours_since_post"], 72.0, atol=2.0)].copy()
    if len(snaps_72) > 0:
        snaps_72["final_value"] = np.log1p(snaps_72["score"] + 2 * snaps_72["descendants"])
        target_info = snaps_72.set_index("story_id")[["snapshot_time", "final_value"]]
        df = df.join(target_info, on="id")
    else:
        df["snapshot_time"] = np.nan
        df["final_value"] = np.nan
        
    df = df.sort_values("time").copy()
    
    def group_expanding_median(group):
        res = np.zeros(len(group))
        times = group["time"].values
        predict_times = group["predict_time"].values
        snap_times = group["snapshot_time"].values
        vals = group["final_value"].values
        
        for i in range(len(group)):
            # Valid past stories: posted before this one, AND their 72h snapshot occurred before this prediction time
            valid_idx = (times[:i] < times[i]) & (snap_times[:i] <= predict_times[i]) & (~np.isnan(vals[:i]))
            if np.any(valid_idx):
                res[i] = np.median(vals[:i][valid_idx])
        return pd.Series(res, index=group.index)
        
    df["author_hist_median"] = df.groupby("by", group_keys=False).apply(group_expanding_median, include_groups=False).fillna(0)
    df["author_post_count"] = df.groupby("by").cumcount()
    df["domain_hist_median"] = df.groupby("domain", group_keys=False).apply(group_expanding_median, include_groups=False).fillna(0)
    
    # 7. Market Context
    df["time_hour_floor"] = df["time"] // 3600
    competing_counts = df.groupby("time_hour_floor").size().to_dict()
    df["competing_stories_hr"] = df["time_hour_floor"].map(competing_counts) - 1
    
    df["market_top30_vel"] = df["v3"].rolling(30, min_periods=1).mean().shift(1).fillna(0)
    
    drop_cols = ["pts_t1", "pts_t2", "pts_t3", "comms_t3", "time_utc", "hour", "dow", 
                 "snapshot_time", "final_value", "predict_time", "time_hour_floor", "story_id"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    from src.eda.helpers import load_or_simulate_data
    
    try:
        stories_df, snapshots_df = load_or_simulate_data(args.db)
    except Exception as e:
        print(f"Data load failed: {e}")
        return
        
    if len(stories_df) == 0:
        print("No stories found.")
        return
        
    for T in [1, 3, 6]:
        print(f"Building features for T={T}...")
        feat_df = build_features(stories_df, snapshots_df, T)
        
        # Prepare targets for the row to be complete
        snaps_72 = snapshots_df[np.isclose(snapshots_df["hours_since_post"], 72.0, atol=2.0)].copy()
        snaps_72["target"] = np.log1p(snaps_72["score"] + 2 * snaps_72["descendants"])
        
        feat_df = feat_df.merge(snaps_72[["story_id", "target"]], left_on="id", right_on="story_id", how="inner")
        feat_df = feat_df.drop(columns=["story_id"])
        
        out_path = out_dir / f"features_t{T}.parquet"
        feat_df.to_parquet(out_path)
        print(f"Wrote {len(feat_df)} rows to {out_path}")

if __name__ == "__main__":
    main()
