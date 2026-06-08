# API Reference

## Core Classes

### FeatureEngineer
**Location:** `src/features/engineer.py`

Main orchestrator for feature engineering.

```python
from src.features.engineer import FeatureEngineer

engineer = FeatureEngineer(config_path='config/metrics/income.yaml')
X_train = engineer.fit_transform(df_train)      # Fit & transform
X_test = engineer.transform(df_test)             # Apply fitted params
names = engineer.get_feature_names()             # Feature names
```

**Methods:**
- `fit_transform(df) → np.ndarray` - Fit scaler and transform
- `transform(df) → np.ndarray` - Apply fitted transformation
- `get_feature_names() → list[str]` - Get feature names
- `get_feature_dimensions() → dict` - Breakdown by module

### CascadingEnsemble
**Location:** `src/models/ensemble.py`

Two-layer ensemble with weight learning.

```python
from src.models.ensemble import CascadingEnsemble

ensemble = CascadingEnsemble(
    catboost_params={...},
    lightgbm_params={...}
)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
alpha, beta = ensemble.get_weights()
```

**Methods:**
- `fit(X, y, X_val=None, y_val=None)` - Train both layers
- `predict(X) → np.ndarray` - Final predictions
- `predict_layer1(X) → np.ndarray` - Layer 1 only
- `predict_layer2(X) → np.ndarray` - Layer 2 only
- `learn_weights(X_val, y_val)` - Optimize fusion weights
- `set_weights(alpha, beta)` - Manual weight setting
- `get_weights() → tuple[float, float]` - Current weights

### HyperparameterOptimizer
**Location:** `src/optimization/hyperparameter.py`

Optuna-based optimization.

```python
from src.optimization.hyperparameter import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    metric_name='income',
    n_trials=100,
    timeout=3600
)

best_params = optimizer.optimize(X_train, y_train)
best_trial = optimizer.study.best_trial
```

**Methods:**
- `optimize(X, y, X_val=None, y_val=None) → dict`
- `get_best_params() → dict`
- `get_study() → optuna.Study`
- `save_results(path: str)`

### DataLoader
**Location:** `src/data/loader.py`

Load raw data from Excel/CSV.

```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_excel('data/raw/game_metrics.xlsx', sheet_name='sheet1')
df = loader.load_csv('data/raw/metrics.csv', date_column='datetime')
```

**Methods:**
- `load_excel(path, sheet_name) → pd.DataFrame`
- `load_csv(path, date_column) → pd.DataFrame`
- `validate_data() → bool`

### TimeSeriesSplitter
**Location:** `src/data/splitter.py`

Time-series aware cross-validation.

```python
from src.data.splitter import TimeSeriesSplitter

splitter = TimeSeriesSplitter(n_splits=5, test_size=0.1, gap=24)

for train_idx, test_idx in splitter.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

**Methods:**
- `split(X) → Iterator[(train_idx, test_idx)]`
- `train_test_split(X, y) → (X_train, X_test, y_train, y_test)`

### MetricsEvaluator
**Location:** `src/evaluation/metrics.py`

Comprehensive evaluation metrics.

```python
from src.evaluation.metrics import MetricsEvaluator

evaluator = MetricsEvaluator()
metrics = evaluator.evaluate(y_true, y_pred)
r2 = evaluator.r2_score(y_true, y_pred)
rmse = evaluator.rmse(y_true, y_pred)
```

**Methods:**
- `evaluate(y_true, y_pred) → dict` - All metrics
- `r2_score(y_true, y_pred) → float`
- `rmse(y_true, y_pred) → float`
- `mae(y_true, y_pred) → float`
- `smape(y_true, y_pred) → float`
- `nrmse(y_true, y_pred) → float`

### ResidualAnalyzer
**Location:** `src/evaluation/residuals.py`

Residual diagnostics.

```python
from src.evaluation.residuals import ResidualAnalyzer

analyzer = ResidualAnalyzer()
diagnostics = analyzer.diagnose(residuals)
ljung_box = analyzer.ljung_box_test(residuals)
```

**Methods:**
- `diagnose(residuals) → dict` - Full diagnostic report
- `ljung_box_test(residuals) → dict` - Autocorrelation
- `normality_test(residuals) → dict` - Normality
- `stationarity_test(residuals) → dict` - Stationarity

## Feature Engineering Modules

### TemporalFeatures
```python
from src.features.temporal import TemporalFeatures
temp = TemporalFeatures()
X = temp.transform(df)  # (n_samples, 10)
```
**Output:** 10 dimensions - hour, day, month, year, etc.

### LagFeatures
```python
from src.features.lag_features import LagFeatures
lag = LagFeatures(windows=[1, 3, 7, 12, 24])
X = lag.transform(df)  # (n_samples, 25)
```
**Output:** 25 dimensions (5 windows × 5 metrics)

### RollingStatistics
```python
from src.features.rolling_stats import RollingStatistics
rolling = RollingStatistics(windows=[3, 7, 12, 24])
X = rolling.transform(df)  # (n_samples, 20)
```
**Output:** 20 dimensions (sum, mean, variance, min, max)

### CyclicalFeatures
```python
from src.features.cyclical import CyclicalFeatures
cyclical = CyclicalFeatures()
X = cyclical.transform(df)  # (n_samples, 10)
```
**Output:** 10 dimensions (sin/cos for 5 cyclic features)

### HolidayFeatures
```python
from src.features.holidays import HolidayFeatures
holidays = HolidayFeatures(config_path='config/features/holiday_calendar.yaml')
X = holidays.transform(df)  # (n_samples, 25)
```
**Output:** 25 dimensions (25 Chinese holidays)

### InteractionFeatures
```python
from src.features.interactions import InteractionFeatures
interactions = InteractionFeatures(base_features=[...])
X = interactions.transform(df)  # (n_samples, 10)
```
**Output:** 10 dimensions (selected interactions)

## Example: Complete Workflow

```python
from src.data.loader import DataLoader
from src.features.engineer import FeatureEngineer
from src.data.splitter import TimeSeriesSplitter
from src.models.ensemble import CascadingEnsemble
from src.evaluation.metrics import MetricsEvaluator

# 1. Load data
loader = DataLoader()
df = loader.load_excel('data/raw/game_metrics.xlsx')

# 2. Engineer features
engineer = FeatureEngineer('config/metrics/income.yaml')
X = engineer.fit_transform(df)
y = df['income'].values

# 3. Split data
splitter = TimeSeriesSplitter(n_splits=5)
train_idx, test_idx = next(splitter.split(X))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 4. Train ensemble
ensemble = CascadingEnsemble(catboost_params={...}, lightgbm_params={...})
ensemble.fit(X_train, y_train)

# 5. Evaluate
evaluator = MetricsEvaluator()
y_pred = ensemble.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)
print(f"Test R²: {metrics['r2']:.4f}")
```

## Utility Functions

```python
from src.utils.io import load_config, save_model
from src.utils.logger import get_logger

config = load_config('config/metrics/income.yaml')
save_model(model, 'models/catboost_income.pkl')
logger = get_logger(__name__)
```

## Next Steps

- See [FEATURES.md](FEATURES.md) for feature engineering details
- See [MODELS.md](MODELS.md) for model configuration
- See [OPTIMIZATION.md](OPTIMIZATION.md) for hyperparameter tuning
- See [WEIGHTS.md](WEIGHTS.md) for weight learning strategy
