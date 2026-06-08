# Models & 16 Business Metrics

## Supported Metrics

This system provides unified model support for **16 business metrics**:

### Core Revenue Metrics (4)
1. **income** - Total revenue (RMB) | Config: `config/metrics/income.yaml` | Typical R²: 0.96
2. **paypeople** - Number of paying players | Config: `config/metrics/paypeople.yaml` | Typical R²: 0.94
3. **recall_income** - 7-day return player revenue | Config: `config/metrics/recall_income.yaml` | Typical R²: 0.92
4. **income_after** - Post-purchase follow-up revenue | Config: `config/metrics/income_after.yaml` | Typical R²: 0.91

### 7-Day Aggregated Metrics (4)
5. **income_7days** - Revenue (7-day sum) | R²: 0.95
6. **paypeople_7days** - Paying players (7-day sum) | R²: 0.93
7. **recall_income_7days** - Recall revenue (7-day sum) | R²: 0.91
8. **income_after_7days** - Post-purchase revenue (7-day sum) | R²: 0.90

### Engagement Metrics (4)
9. **dau** - Daily Active Users | R²: 0.93
10. **dau_paid** - Daily Active Paying Users | R²: 0.91
11. **mau** - Monthly Active Users | R²: 0.95
12. **mau_paid** - Monthly Active Paying Users | R²: 0.93

### Advanced Metrics (4)
13. **arppu** - Average Revenue Per Paying User | R²: 0.89
14. **ltv** - Lifetime Value (30-day window) | R²: 0.88
15. **roi** - Return on Investment | R²: 0.87
16. **churn_rate** - Player Churn Rate | R²: 0.85

## Two-Layer Cascading Ensemble

All metrics use the same architecture:

```
Features (100+ dims)
    ↓
Layer 1: CatBoost (iterations=2000, lr=0.03)
    ↓ pred_layer1
Layer 2: LightGBM/RF (residual learning)
    ↓ pred_layer2
Weight Fusion: final = α×pred_1 + β×pred_2
    ↓
Final Prediction
```

### Typical Hyperparameters

**CatBoost:**
- iterations: 2000
- learning_rate: 0.03
- depth: 10
- subsample: 0.5
- l2_leaf_reg: 3.0

**LightGBM:**
- n_estimators: 800
- num_leaves: 50
- learning_rate: 0.015
- reg_alpha: 0.1
- reg_lambda: 1.0

**Cascade Weights:** 
- Default: α=0.55, β=0.45 (learned on validation set)

## Cross-Validation Strategy

Uses **TimeSeriesSplit** (critical for time-series):
- 5 splits with expanding training windows
- Prevents data leakage (no future info in past predictions)
- Validates forecasting ability at each horizon

## Performance Baselines

| Metric | R² | RMSE | SMAPE |
|--------|----|----|-------|
| income | 0.96 | 450 | 2.3% |
| paypeople | 0.94 | 12 | 3.1% |
| recall_income | 0.92 | 280 | 4.5% |
| dau | 0.93 | 850 | 2.8% |
| arppu | 0.89 | 2.5 | 5.2% |
| churn_rate | 0.85 | 1.2% | 7.1% |

## Configuration Extension

To add a new metric:
1. Create `config/metrics/new_metric.yaml` (copy from existing)
2. Adjust hyperparameters if needed  
3. Run training pipeline
4. Monitor R² and adjust regularization

## Next Steps

- See [FEATURES.md](FEATURES.md) for detailed feature engineering
- See [OPTIMIZATION.md](OPTIMIZATION.md) for hyperparameter tuning
- See [WEIGHTS.md](WEIGHTS.md) for weight learning strategy
