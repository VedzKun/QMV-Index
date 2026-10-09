"""
helpers.py — Data loading, simulation, and plotting logic for EDA.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from src.collect.db import DB_DEFAULT

logger = logging.getLogger(__name__)


def _simulate_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic HN data if the DB is empty (for the portfolio)."""
    np.random.seed(42)
    n_stories = 5000

    times = np.random.uniform(1700000000, 1710000000, n_stories)
    hours = (times % 86400) / 3600
    dows = (times // 86400) % 7
    
    quality = np.random.lognormal(mean=0, sigma=1.5, size=n_stories)
    
    archetype = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1], size=n_stories)
    quality[archetype == 0] *= 0.2
    quality[archetype == 2] *= 5.0
    
    season_boost = np.where((hours >= 12) & (hours <= 16), 1.5, 1.0)
    season_boost = np.where(dows >= 5, season_boost * 0.7, season_boost)
    quality *= season_boost

    stories_df = pd.DataFrame({
        "id": np.arange(1, n_stories + 1),
        "title": [f"Synthetic Story {i}" for i in range(1, n_stories + 1)],
        "url": [f"https://example.com/{i}" if i % 10 != 0 else "" for i in range(1, n_stories + 1)],
        "by": [f"user_{i%500}" for i in range(n_stories)],
        "time": times,
        "type": "story",
        "is_dead": np.random.binomial(1, 0.02, n_stories),
        "is_deleted": np.random.binomial(1, 0.01, n_stories),
        "archetype": archetype,
        "quality": quality
    })
    
    target_hours = [1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
    snap_records = []
    
    for _, row in stories_df.iterrows():
        sid = row["id"]
        arch = row["archetype"]
        q = row["quality"]
        
        for h in target_hours:
            if arch == 0:
                growth = np.log1p(h) / np.log1p(72)
            elif arch == 1:
                growth = (h / 72) ** 1.5
            else:
                growth = 1.0 - np.exp(-h / 6.0)
                
            current_q = q * growth
            current_q *= np.random.lognormal(mean=0, sigma=0.2)
            
            in_top = 0
            if arch == 2 and h >= 3.0:
                current_q *= 2.5
                in_top = 1
                
            pts = int(current_q * 10)
            comms = int(current_q * 3)
            
            snap_records.append({
                "story_id": sid,
                "snapshot_time": row["time"] + h * 3600,
                "hours_since_post": h,
                "score": pts,
                "descendants": comms,
                "is_dead": row["is_dead"],
                "is_deleted": row["is_deleted"],
                "in_top_500": in_top
            })
            
    snapshots_df = pd.DataFrame(snap_records)
    return stories_df.drop(columns=["archetype", "quality"]), snapshots_df


def load_or_simulate_data(db_path: str = str(DB_DEFAULT)) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not Path(db_path).exists():
        logger.info("DB not found, simulating data...")
        return _simulate_data()
        
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        if count < 1000:
            logger.info("Insufficient DB data, simulating data...")
            return _simulate_data()
            
        stories = pd.read_sql("SELECT * FROM stories", conn)
        snapshots = pd.read_sql("SELECT * FROM snapshots", conn)
        
        if "in_top_500" not in snapshots.columns:
            snapshots["in_top_500"] = (snapshots["score"] > 50).astype(int)
            
        return stories, snapshots
    finally:
        conn.close()


def prepare_target(snapshots_df: pd.DataFrame) -> pd.DataFrame:
    df_72 = snapshots_df[np.isclose(snapshots_df["hours_since_post"], 72.0, atol=2.0)].copy()
    df_72["target_raw"] = df_72["score"] + 2 * df_72["descendants"]
    df_72["target_log"] = np.log1p(df_72["target_raw"])
    return df_72


def plot_distribution(df_72: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(df_72["target_raw"], bins=50, ax=axes[0])
    axes[0].set_title("Raw Realized Value at 72h")
    
    sns.histplot(df_72["target_log"], bins=50, ax=axes[1])
    axes[1].set_title("Log Realized Value (Target)")
    
    total_val = df_72["target_raw"].sum()
    top_1_val = df_72.nlargest(int(len(df_72) * 0.01), "target_raw")["target_raw"].sum()
    top_10_val = df_72.nlargest(int(len(df_72) * 0.10), "target_raw")["target_raw"].sum()
    
    print(f"Top 1% captures {top_1_val/total_val:.1%} of attention.")
    print(f"Top 10% captures {top_10_val/total_val:.1%} of attention.")
    plt.tight_layout()
    plt.show()


def plot_trajectories_and_clusters(snapshots_df: pd.DataFrame):
    horizons = [1.0, 3.0, 6.0, 12.0, 24.0, 48.0, 72.0]
    df = snapshots_df[np.isclose(snapshots_df["hours_since_post"].values[:, None], horizons, atol=0.5).any(axis=1)].copy()
    df["horizon"] = df["hours_since_post"].round()
    
    traj = df.pivot_table(index="story_id", columns="horizon", values="score").dropna()
    if len(traj) < 10: return
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    median, p25, p75 = traj.median(), traj.quantile(0.25), traj.quantile(0.75)
    axes[0].plot(median.index, median.values, 'b-', label="Median")
    axes[0].fill_between(median.index, p25, p75, color='b', alpha=0.2, label="IQR")
    axes[0].set_title("Overall Trajectory")
    axes[0].legend()
    
    traj_norm = traj.div(traj[72.0].clip(lower=1), axis=0)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(traj_norm)
    
    archetypes = ["Fast-burn / Viral", "Slow-build", "Flat / Plateau"]
    order = np.argsort(-kmeans.cluster_centers_[:, 1])
    
    for i, c in enumerate(order):
        if i >= len(archetypes): break
        cluster_data = traj_norm[clusters == c]
        axes[1].plot(traj_norm.columns, cluster_data.median(), label=f"{archetypes[i]} (n={len(cluster_data)})")
        
    axes[1].set_title("Trajectory Archetypes")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_early_vs_final(snapshots_df: pd.DataFrame, df_72: pd.DataFrame):
    horizons = [1.0, 3.0, 6.0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, h in enumerate(horizons):
        df_h = snapshots_df[np.isclose(snapshots_df["hours_since_post"], h, atol=0.5)].copy()
        df_h["early_val"] = np.log1p(df_h["score"] + 2 * df_h["descendants"])
        merged = df_h.merge(df_72[["story_id", "target_log"]], on="story_id").dropna()
        if len(merged) < 2: continue
            
        sns.regplot(x="early_val", y="target_log", data=merged, ax=axes[i], 
                    scatter_kws={"alpha":0.2, "s":10}, line_kws={"color":"red"})
        corr = merged["early_val"].corr(merged["target_log"])
        r2 = r2_score(merged["target_log"], merged["early_val"])
        axes[i].set_title(f"T={h}h vs 72h\nCorr: {corr:.2f} | R2: {r2:.2f}")
        
    plt.tight_layout()
    plt.show()


def plot_seasonality(stories_df: pd.DataFrame, df_72: pd.DataFrame):
    merged = stories_df.merge(df_72[["story_id", "target_log"]], left_on="id", right_on="story_id")
    merged["time_utc"] = pd.to_datetime(merged["time"], unit="s", utc=True)
    merged["hour"], merged["dow"] = merged["time_utc"].dt.hour, merged["time_utc"].dt.dayofweek
    
    heatmap_data = merged.pivot_table(values="target_log", index="dow", columns="hour", aggfunc="mean")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_data, cmap="viridis", yticklabels=days)
    plt.title("Mean 72h Target by Post Time (UTC)")
    plt.tight_layout()
    plt.show()


def plot_front_page_effect(snapshots_df: pd.DataFrame):
    horizons = [1.0, 3.0, 6.0]
    df = snapshots_df[np.isclose(snapshots_df["hours_since_post"].values[:, None], horizons, atol=0.5)].copy()
    df["horizon"] = df["hours_since_post"].round()
    
    piv_score = df.pivot_table(index="story_id", columns="horizon", values="score")
    piv_top = df.pivot_table(index="story_id", columns="horizon", values="in_top_500")
    
    valid = piv_score.dropna().index.intersection(piv_top.dropna().index)
    piv_score, piv_top = piv_score.loc[valid], piv_top.loc[valid]
    
    v_1_to_3 = (piv_score[3.0] - piv_score[1.0]) / 2.0
    v_3_to_6 = (piv_score[6.0] - piv_score[3.0]) / 3.0
    
    transitioned = (piv_top[1.0] == 0) & (piv_top[3.0] == 1)
    stayed_off = (piv_top[1.0] == 0) & (piv_top[3.0] == 0)
    
    plt.figure(figsize=(6, 4))
    data = [
        (v_3_to_6[stayed_off] - v_1_to_3[stayed_off]).clip(-10, 50),
        (v_3_to_6[transitioned] - v_1_to_3[transitioned]).clip(-10, 50)
    ]
    plt.boxplot(data, labels=["Stayed Off Front Page", "Hit Front Page at 3h"])
    plt.title("Change in Velocity (Δ Score/Hour) after T=3h")
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_missingness_and_outliers(stories_df: pd.DataFrame, snapshots_df: pd.DataFrame):
    print("--- Data Quality Audit ---")
    n_stories, n_dead, n_deleted = len(stories_df), stories_df["is_dead"].sum(), stories_df["is_deleted"].sum()
    print(f"Total stories: {n_stories}")
    print(f"Moderated: {n_dead} dead ({n_dead/n_stories:.1%}), {n_deleted} deleted ({n_deleted/n_stories:.1%})")
    
    empty_urls = (stories_df["url"] == "").sum()
    dup_urls = stories_df[stories_df["url"] != ""].duplicated(subset=["url"]).sum()
    print(f"Text-only posts (empty URL): {empty_urls} ({empty_urls/n_stories:.1%})")
    print(f"Duplicate URLs (reposts): {dup_urls}")
    
    df_72 = prepare_target(snapshots_df)
    zeros = (df_72["target_raw"] == 0).sum()
    print(f"Zero engagement by 72h: {zeros} stories ({zeros/len(df_72):.1%})")
