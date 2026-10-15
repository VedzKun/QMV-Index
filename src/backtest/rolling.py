import pandas as pd
import numpy as np
from src.models.metrics import calculate_metrics

def rolling_origin_cv(df, model, feature_cols, target_col="target", 
                      initial_train_days=30, step_days=15, gap_hours=72):
    times = df["time"].values
    min_t, max_t = times.min(), times.max()
    
    initial_train_sec = initial_train_days * 24 * 3600
    step_sec = step_days * 24 * 3600
    gap_sec = gap_hours * 3600
    
    results = []
    
    train_end = min_t + initial_train_sec
    fold = 0
    
    while True:
        val_start = train_end + gap_sec
        val_end = val_start + step_sec
        
        if val_start >= max_t:
            break
            
        train_idx = np.where(times <= train_end)[0]
        val_idx = np.where((times >= val_start) & (times < val_end))[0]
        
        if len(train_idx) > 0 and len(val_idx) > 0:
            X_train = np.array(df.iloc[train_idx][feature_cols].values, dtype=np.float64, copy=True, order='C')
            y_train = np.array(df.iloc[train_idx][target_col].values, dtype=np.float64, copy=True, order='C')
            X_val = np.array(df.iloc[val_idx][feature_cols].values, dtype=np.float64, copy=True, order='C')
            y_val = df.iloc[val_idx][target_col].values
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            metrics = calculate_metrics(y_val, preds)
            metrics["fold"] = fold
            metrics["train_end"] = pd.to_datetime(train_end, unit='s')
            results.append(metrics)
            
        train_end += step_sec
        fold += 1
        
    return pd.DataFrame(results)
