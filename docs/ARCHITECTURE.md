# GameRevenueForecast - Architecture Overview

## System Design

This project implements a two-layer cascading ensemble for game revenue forecasting.

### Layer 1: CatBoost (Base Predictor)
- Direct prediction from features
- 2000 iterations, learning_rate=0.03, depth=10

### Layer 2: LightGBM/RandomForest (Residual Learner)  
- Predicts residuals from Layer 1
- Learns what Layer 1 misses

### Fusion: Learnable Weights
- final = α * pred_cat + β * pred_residual
- α and β learned via validation set optimization
- Improvement over fixed 1:1 combination

## Module Structure

- src/data/: Data loading and preprocessing
- src/features/: 100+ feature engineering (temporal, lag, rolling, cyclical, holidays)
- src/models/: Cascading ensemble architecture
- src/optimization/: Hyperparameter tuning with Optuna
- src/evaluation/: Metrics, diagnostics, visualization
- src/config/: Configuration management
- src/utils/: Utilities and helpers

## Data Pipeline

Raw Data → Preprocessing → Feature Engineering → Scaling → 
Layer 1 (CatBoost) → Residuals → Layer 2 (LightGBM) → 
Weight Optimization → Ensemble Predictions

## Supported Metrics

16 business indicators:
- income, paypeople, recall_income, income_after
- income_paytime, income_paytime_after
- recall_income_paytime, recall_income_paytime_after
- payment_hour, paypeople_paytime, recall_paypeople_paytime
- recall_paypeople, regCount, recall_regCount
- allLoginCount, allIncomeAfter
