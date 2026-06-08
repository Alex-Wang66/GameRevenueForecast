# Hyperparameter Optimization Guide

## Overview

This guide covers the **Optuna-based hyperparameter optimization** system that automatically tunes model hyperparameters for maximum predictive performance using Bayesian optimization.

## Why Optuna?

- **Bayesian Optimization:** Uses Tree-structured Parzen Estimator (TPE) for intelligent search
- **Fast:** ~10-20% faster convergence than Grid Search or Random Search
- **Pruning:** Stops unpromising trials early
- **Reproducibility:** Seed support for deterministic results
- **Parallelization:** Supports distributed optimization across multiple machines

## System Architecture

```
Training Data
    ↓
Optuna Loop (100 trials):
  ├─ Sample hyperparameters
  ├─ Train model
  ├─ Evaluate with TimeSeriesSplit CV
  ├─ Prune if needed
  └─ Record score
    ↓
Best Trial: hyperparameters + best R²
    ↓
Retrain final model with best hyperparameters
```

## Configuration

```yaml
optimization:
  method: optuna
  direction: maximize         # Maximize R² score
  n_trials: 100              # Number of optimization trials
  timeout: 3600              # 1 hour timeout
  sampler: TPE               # Tree-structured Parzen Estimator
  seed: 42                   # Reproducibility
  
  pruner:
    type: median             # Stop if below median
    n_startup_trials: 10     # Skip first 10 trials
    n_warmup_steps: 30       # Warmup before pruning
```

## Search Space

**CatBoost hyperparameters:**
- learning_rate: [0.01, 0.05]
- depth: [8, 15]
- subsample: [0.3, 0.8]
- l2_leaf_reg: [0.1, 5.0]

**LightGBM hyperparameters:**
- learning_rate: [0.005, 0.03]
- num_leaves: [31, 127]
- max_depth: [8, 15]
- reg_alpha: [0.0, 0.5]
- reg_lambda: [0.5, 2.0]

## Example Results

**Sample optimization run for "income" metric:**

```
Best trial:
  Value: 0.9601
  Trial #: 87
  Params:
    cb_lr: 0.0312
    cb_depth: 11
    lgb_lr: 0.0168
    lgb_leaves: 63
    ... (8 more hyperparams)

Summary:
  Total Trials: 100
  Median CV R²: 0.9450
  Std Dev: 0.0045
  Pruned Trials: 12
  Time Elapsed: 58 minutes
```

## Optimization Workflow

```python
from src.optimization.hyperparameter import HyperparameterOptimizer

# Initialize
optimizer = HyperparameterOptimizer(
    metric_name='income',
    n_trials=100,
    timeout=3600
)

# Run optimization
best_params = optimizer.optimize(X_train, y_train)

# Get results
print(f"Best R²: {optimizer.study.best_value:.4f}")
print(f"Best params: {optimizer.study.best_trial.params}")

# Save for later use
optimizer.save_results('logs/optimization/income_results.pkl')
```

## Tips for Tuning

1. **Search Space Bounds:** Use informed ranges based on domain knowledge
2. **Trial Budget:** 
   - Quick: 20 trials, 5 min
   - Thorough: 100 trials, 1 hour
   - Extensive: 500 trials, 4 hours
3. **Parallelization:** Run multiple trials in parallel with `n_jobs=8`
4. **Monitoring:** Track progress with callbacks and intermediate reporting

## Hyperparameter Importance

After optimization, analyze which hyperparameters matter most:

```python
from optuna.importance import get_param_importances

importances = get_param_importances(optimizer.study)

for param_name, importance in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"{param_name}: {importance:.4f}")
```

## Per-Metric Optimization

Run optimization for each of the 16 metrics:

```bash
python examples/hyperparameter_search.py --metric income
python examples/hyperparameter_search.py --metric paypeople
python examples/hyperparameter_search.py --metric recall_income
# ... etc for all 16 metrics
```

Results saved to `logs/optimization/[metric]_best_params.json`

## Next Steps

- See [MODELS.md](MODELS.md) for metric-specific configurations
- See [WEIGHTS.md](WEIGHTS.md) for weight learning after optimization
- See [FEATURES.md](FEATURES.md) for feature engineering
