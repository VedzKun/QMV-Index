# Feature Dictionary

## 1. Velocity
* **vel_w1**: Points per hour in window [0, T/3].
  * **Why it matters**: Captures immediate initial traction.
  * **Leakage risk**: Low, uses strictly past snapshots relative to T.
* **vel_w2**: Points per hour in window [T/3, 2T/3].
  * **Why it matters**: Captures mid-term sustained interest.
  * **Leakage risk**: Low, uses strictly past snapshots.
* **vel_w3**: Points per hour in window [2T/3, T].
  * **Why it matters**: Captures latest momentum before prediction.
  * **Leakage risk**: Low, uses strictly past snapshots.
* **accel**: Difference between latest and earliest velocity (`vel_w3 - vel_w1`).
  * **Why it matters**: Identifies stories that are picking up speed vs dying out.
  * **Leakage risk**: Low, uses strictly past snapshots.
* **late_early_ratio**: Ratio of `vel_w3` to `vel_w1`.
  * **Why it matters**: Relative momentum indicator.
  * **Leakage risk**: Low.

## 2. Engagement Mix
* **c2p_ratio**: Ratio of comments to points at time T.
  * **Why it matters**: High ratio indicates controversial or highly conversational topics.
  * **Leakage risk**: Low.
* **comment_vel**: Comments per hour up to time T.
  * **Why it matters**: Identifies fast-moving discussions.
  * **Leakage risk**: Low.
* **comment_score_lag**: Time (in hours) to the first comment.
  * **Why it matters**: Shorter lag indicates immediately engaging content.
  * **Leakage risk**: Low.

## 3. Recency and Timing
* **hour_sin** / **hour_cos**: Cyclical encoding of the hour of the day.
  * **Why it matters**: Traffic varies predictably by time of day.
  * **Leakage risk**: None.
* **dow_sin** / **dow_cos**: Cyclical encoding of the day of the week.
  * **Why it matters**: Weekend traffic behaves differently from weekday traffic.
  * **Leakage risk**: None.
* **is_weekend**: Boolean flag for Saturday/Sunday.
  * **Why it matters**: Clear distinct traffic pattern for weekends.
  * **Leakage risk**: None.

## 4. Content
* **title_len**: Length of the title in characters.
  * **Why it matters**: Title brevity or descriptiveness can impact CTR.
  * **Leakage risk**: None.
* **has_number**: Presence of numbers in the title.
  * **Why it matters**: Listicles and data-driven posts often perform well.
  * **Leakage risk**: None.
* **has_question**: Presence of a question mark in the title.
  * **Why it matters**: Engages reader curiosity.
  * **Leakage risk**: None.
* **is_show_hn**: Flag for "Show HN" posts.
  * **Why it matters**: Show HN has a dedicated audience and voting behavior.
  * **Leakage risk**: None.
* **is_ask_hn**: Flag for "Ask HN" posts.
  * **Why it matters**: Ask HN often drives higher comment-to-point ratios.
  * **Leakage risk**: None.
* **title_svd_[0-19]**: First 20 SVD components of the title TF-IDF matrix.
  * **Why it matters**: Captures latent semantic topics in titles.
  * **Leakage risk**: Moderate (fit over all stories could leak future terminology, strictly should be fit on train only).

## 5. Author Quality
* **author_quality**: Historical median final outcome over the author's PRIOR stories.
  * **Why it matters**: Captures an author's historical success rate (reputation).
  * **Leakage risk**: High if not computed correctly. Handled via expanding window strictly shifted to exclude the current story.
* **author_post_count**: Number of previous posts by the author.
  * **Why it matters**: Indicates author experience and community familiarity.
  * **Leakage risk**: Low, uses strictly prior counts.

## 6. Domain Reputation
* **domain_rep**: Expanding-window median outcome per domain for prior stories.
  * **Why it matters**: Certain domains (e.g., github.com, nytimes.com) have high inherent trust or interest.
  * **Leakage risk**: High if not computed correctly. Handled via expanding window strictly shifted to exclude the current story.

## 7. Market Context
* **competing_stories_same_hour**: Number of other stories posted in the same hour.
  * **Why it matters**: High competition can bury a good story.
  * **Leakage risk**: Low (uses creation times, which are known).
* **top30_avg_vel**: Average velocity of the top 30 stories on the site at prediction time.
  * **Why it matters**: Captures macro site traffic/activity levels.
  * **Leakage risk**: Moderate. Must be strictly computed using snapshots available at or before T hours after the current story's creation.
