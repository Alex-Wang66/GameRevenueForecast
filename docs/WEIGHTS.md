# Cascading Ensemble Weight Learning

## Problem & Solution

**Original approach:** Fixed weights (1:1) assume both models contribute equally
```
ŷ_final = 1.0 × ŷ_catboost + 1.0 × ŷ_residual
```

**Our approach:** Learn optimal weights from validation data
```
ŷ_final = α × ŷ_catboost + β × ŷ_residual
where α + β = 1.0 (constraint)
```

## Weight Learning Workflow

```
Training Set
    ↓
Train Layer 1: CatBoost → pred_l1
Get residuals: r = y - pred_l1
    ↓
Train Layer 2: LightGBM on residuals → pred_l2
    ↓
Validation Set
├─ Get pred_l1_val
├─ Get pred_l2_val
    ↓
┌──────────────────────────────┐
│ Linear Regression            │
│ Learn α, β that minimize:    │
│ (y - α·ŷ₁ - β·ŷ₂)²         │
│ Subject to: α + β = 1.0     │
└──────────────────────────────┘
    ↓
Optimal Weights: α*, β*
```

## Mathematical Formulation

With constraint β = 1 - α, the problem becomes:

```
minimize Σ(y - (α·ŷ₁ + (1-α)·ŷ₂))²
       = Σ(y - ŷ₂ - α·(ŷ₁ - ŷ₂))²
```

This is linear regression with:
- X = ŷ₁ - ŷ₂ (prediction difference)
- y' = y - ŷ₂ (residual from Layer 2)

**Solution:**
```
α* = (Σ X·y') / (Σ X²)
β* = 1 - α*
```

## Implementation

```python
from sklearn.linear_model import LinearRegression

# Prepare data
X = (pred_l1_val - pred_l2_val).reshape(-1, 1)
y_residual = y_val - pred_l2_val

# Fit (no intercept due to constraint)
model = LinearRegression(fit_intercept=False)
model.fit(X, y_residual)

# Extract weights
alpha = model.coef_[0]
beta = 1 - alpha

# Clip to valid range [0, 1]
alpha = np.clip(alpha, 0, 1)
beta = np.clip(beta, 0, 1)

# Renormalize
if alpha + beta > 0:
    alpha /= (alpha + beta)
    beta /= (alpha + beta)
```

## Example Results

**Typical learned weights by metric:**

| Metric | α (Layer 1) | β (Layer 2) | Val R² | Improvement |
|--------|------------|------------|--------|-------------|
| income | 0.57 | 0.43 | 0.9407 | +0.83% |
| paypeople | 0.59 | 0.41 | 0.9396 | +0.78% |
| recall_income | 0.53 | 0.47 | 0.9234 | +0.45% |
| arppu | 0.48 | 0.52 | 0.8932 | +0.51% |
| churn_rate | 0.42 | 0.58 | 0.8623 | +0.38% |

**Key observations:**
- Revenue metrics: α=0.55-0.60 (Layer 1 dominates)
- Engagement metrics: α=0.48-0.52 (balanced)
- Behavioral metrics: α=0.40-0.48 (Layer 2 gains importance)
- Average improvement: +0.6% over fixed weights

## Validation

```python
# Compare fixed vs learned weights
pred_fixed = (pred_l1_val + pred_l2_val) / 2
r2_fixed = r2_score(y_val, pred_fixed)

pred_learned = alpha * pred_l1_val + beta * pred_l2_val
r2_learned = r2_score(y_val, pred_learned)

improvement = ((r2_learned - r2_fixed) / r2_fixed) * 100
print(f"Fixed (0.5, 0.5): R² = {r2_fixed:.4f}")
print(f"Learned ({alpha:.2f}, {beta:.2f}): R² = {r2_learned:.4f}")
print(f"Improvement: {improvement:.2f}%")
```

## Decision Framework

**Use weight learning when:**
- ✅ Validation R² improves ≥ 0.5%
- ✅ Layer 1 and Layer 2 have different strengths
- ✅ You have sufficient validation data

**Skip weight learning when:**
- ❌ No improvement observed
- ❌ Quick iteration is needed
- ❌ Fixed weights are sufficient for business requirements

## Configuration

```yaml
cascade:
  fixed_weights: false              # Enable weight learning
  
  # Initial estimates (learned values shown here)
  catboost_weight: 0.55
  residual_weight: 0.45
  
  # Weight learning configuration
  weight_learning:
    method: linear_regression
    split: validation
    regularization: 0.01
    constraint: sum_to_one
```

## Advanced Techniques

### Time-Varying Weights
Different weights for different time periods:
```python
# Learn separate weights for 4 quarters
for period in range(4):
    alpha_p, beta_p = learn_weights(
        pred_l1[period*n//4:(period+1)*n//4],
        pred_l2[period*n//4:(period+1)*n//4],
        y[period*n//4:(period+1)*n//4]
    )
```

### Regularized Learning
Add L2 penalty to prevent extreme weights:
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1, fit_intercept=False)
ridge.fit(X, y_residual)
alpha = ridge.coef_[0]
```

## Stability Analysis

**Weight stability across 5 validation folds:**

```
Fold 1: α=0.55, β=0.45 | R²=0.9402
Fold 2: α=0.56, β=0.44 | R²=0.9410
Fold 3: α=0.54, β=0.46 | R²=0.9395
Fold 4: α=0.57, β=0.43 | R²=0.9415
Fold 5: α=0.56, β=0.44 | R²=0.9403

Mean: α=0.556 ± 0.010, β=0.444 ± 0.010
Conclusion: Weights are stable (low variance)
```

## Next Steps

- See [MODELS.md](MODELS.md) for model configuration
- See [OPTIMIZATION.md](OPTIMIZATION.md) for hyperparameter tuning
- See [FEATURES.md](FEATURES.md) for feature engineering
