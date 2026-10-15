import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

def compute_qmv_index(df, pred_col="pred_value", time_col="time", window_days=30):
    df = df.sort_values(time_col).copy()
    qmv = np.zeros(len(df))
    
    times = df[time_col].values
    preds = df[pred_col].values
    window_sec = window_days * 24 * 3600
    
    for i in range(len(df)):
        current_time = times[i]
        start_time = current_time - window_sec
        
        mask = (times < current_time) & (times >= start_time)
        trailing_preds = preds[mask]
        
        if len(trailing_preds) < 20:
            if i > 0:
                trailing_preds = preds[:i]
            else:
                qmv[i] = 50.0
                continue
                
        qmv[i] = percentileofscore(trailing_preds, preds[i], kind='weak')
        
    df["qmv_index"] = qmv
    return df
