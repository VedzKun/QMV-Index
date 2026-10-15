# Failure Analysis Report

This report analyzes the 20 worst predictions (by absolute error on the log target) from the T=3h HistGradientBoostingRegressor model.

## Summary of Patterns
- **Late Surges (Under-predicted):** Stories where `v3` (velocity at hour 3) was low, but the final score was massive. Often indicates a story hitting the front page *after* T=3h.
- **Duds (Over-predicted):** Stories that had high initial velocity but suddenly stopped growing. Often indicates the story was flagged, dropped off the front page quickly, or was controversial but ultimately downvoted.

## The 20 Worst Predictions

### 1. Synthetic Story 3854 (ID: 3854)
- **Error Type:** Under-predicted (Error: 4.21)
- **Predicted Log Value:** 0.95 | **Actual:** 5.16
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 2. Synthetic Story 4051 (ID: 4051)
- **Error Type:** Under-predicted (Error: 4.09)
- **Predicted Log Value:** 1.06 | **Actual:** 5.15
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 3. Synthetic Story 3507 (ID: 3507)
- **Error Type:** Under-predicted (Error: 4.07)
- **Predicted Log Value:** 0.67 | **Actual:** 4.74
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 4. Synthetic Story 4518 (ID: 4518)
- **Error Type:** Under-predicted (Error: 4.04)
- **Predicted Log Value:** 1.32 | **Actual:** 5.37
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 5. Synthetic Story 4841 (ID: 4841)
- **Error Type:** Under-predicted (Error: 4.01)
- **Predicted Log Value:** 1.29 | **Actual:** 5.30
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 6. Synthetic Story 3823 (ID: 3823)
- **Error Type:** Under-predicted (Error: 3.92)
- **Predicted Log Value:** 0.81 | **Actual:** 4.74
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 7. Synthetic Story 4224 (ID: 4224)
- **Error Type:** Under-predicted (Error: 3.91)
- **Predicted Log Value:** 3.55 | **Actual:** 7.47
- **Velocity (v1->v3):** 1.0 -> 2.5
- **Suspected Reason:** Unknown

### 8. Synthetic Story 4192 (ID: 4192)
- **Error Type:** Under-predicted (Error: 3.89)
- **Predicted Log Value:** 2.13 | **Actual:** 6.02
- **Velocity (v1->v3):** 0.0 -> 0.5
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 9. Synthetic Story 4767 (ID: 4767)
- **Error Type:** Under-predicted (Error: 3.89)
- **Predicted Log Value:** 1.16 | **Actual:** 5.05
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 10. Synthetic Story 141 (ID: 141)
- **Error Type:** Under-predicted (Error: 3.85)
- **Predicted Log Value:** 1.07 | **Actual:** 4.91
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 11. Synthetic Story 2101 (ID: 2101)
- **Error Type:** Under-predicted (Error: 3.83)
- **Predicted Log Value:** 1.98 | **Actual:** 5.81
- **Velocity (v1->v3):** 0.0 -> 0.5
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 12. Synthetic Story 3207 (ID: 3207)
- **Error Type:** Under-predicted (Error: 3.82)
- **Predicted Log Value:** 1.16 | **Actual:** 4.98
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 13. Synthetic Story 4649 (ID: 4649)
- **Error Type:** Under-predicted (Error: 3.81)
- **Predicted Log Value:** 1.24 | **Actual:** 5.06
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 14. Synthetic Story 2854 (ID: 2854)
- **Error Type:** Under-predicted (Error: 3.74)
- **Predicted Log Value:** 1.20 | **Actual:** 4.94
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 15. Synthetic Story 2315 (ID: 2315)
- **Error Type:** Under-predicted (Error: 3.69)
- **Predicted Log Value:** 1.21 | **Actual:** 4.90
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 16. Synthetic Story 2569 (ID: 2569)
- **Error Type:** Under-predicted (Error: 3.60)
- **Predicted Log Value:** 1.42 | **Actual:** 5.02
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 17. Synthetic Story 1445 (ID: 1445)
- **Error Type:** Under-predicted (Error: 3.59)
- **Predicted Log Value:** 1.19 | **Actual:** 4.78
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 18. Synthetic Story 1855 (ID: 1855)
- **Error Type:** Under-predicted (Error: 3.58)
- **Predicted Log Value:** 3.73 | **Actual:** 7.32
- **Velocity (v1->v3):** 1.0 -> 2.5
- **Suspected Reason:** Unknown

### 19. Synthetic Story 3706 (ID: 3706)
- **Error Type:** Under-predicted (Error: 3.58)
- **Predicted Log Value:** 2.47 | **Actual:** 6.05
- **Velocity (v1->v3):** 0.0 -> 0.5
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)

### 20. Synthetic Story 1608 (ID: 1608)
- **Error Type:** Under-predicted (Error: 3.57)
- **Predicted Log Value:** 1.27 | **Actual:** 4.84
- **Velocity (v1->v3):** 0.0 -> 0.0
- **Suspected Reason:** Late Surge (Velocity was low at T=3h but final score is huge)
