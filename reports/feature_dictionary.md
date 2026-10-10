# QMV Index — Feature Dictionary

## 1. Velocity
* **v1, v2, v3**: Score accumulation rate (points per hour) in intervals [0, T/3], [T/3, 2T/3], [2T/3, T]. Captures momentum.
* **acceleration**: `v3 - v1`. Detects stories that are picking up steam vs dying out.
* **late_early_vel_ratio**: `v3 / (v1 + epsilon)`. Similar to acceleration but multiplicative.

## 2. Engagement Mix
* **comments_points_ratio**: Controversial or highly technical posts have higher comment ratios.
* **comment_velocity**: Total comments / T.
* **comment_lag**: `points_at_T - comments_at_T`. 

## 3. Recency and Timing
* **hour, dow**: Raw UTC hour and Day of Week.
* **is_weekend**: Boolean penalty for weekends.
* **hour_sin, hour_cos, dow_sin, dow_cos**: Cyclical encodings to ensure 23:00 and 00:00 are close to each other in vector space.

## 4. Content
* **title_len**: Longer titles sometimes indicate deeper content.
* **has_question**: "Ask HN" or curious titles.
* **has_number**: Listicles or year references.
* **is_ask_hn, is_show_hn**: Explicit tags.
* **domain**: Target extracted from the URL.
* **title_svd_0...19**: TF-IDF vectors reduced via TruncatedSVD to capture semantic topics without exploding dimensionality.

## 5 & 6. Historical Reputation
* **author_hist_median**: The median 72h realized value of all *prior* stories by this author. 
  * *Leakage Prevention*: Uses an expanding window strictly on chronologically sorted stories and shifts the target row by 1 so the current story's outcome is **never** included.
* **author_post_count**: Number of prior posts by this author.
* **domain_hist_median**: Expanding median for the domain (e.g. `github.com` vs `nytimes.com`). Also strictly shifted to avoid leakage.

## 7. Market Context
* **competing_stories_hr**: Count of other stories posted in the exact same hour. High supply = lower average visibility.
* **market_top30_vel**: Average velocity (`v3`) of the 30 stories immediately preceding this one. Proxy for current front-page velocity requirements.
