# GameRevenueForecast 🎮📊

> 游戏变现分时预测系统 | Time-Series Forecasting for Game Monetization

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 项目介绍

**GameRevenueForecast** 是基于4399微信小游戏的生产级游戏变现分时预测系统。采用**二层级联融合集成**架构和**可学习权重融合**，精确预测游戏的16个关键商业指标。

### ✨ 核心特性

- **🤖 二层级联融合** - CatBoost基础预测 + LightGBM/RF残差学习，支持可学习权重
- **📊 100+维特征工程** - 8个特征模块（滞后、滚动统计、循环编码、节假日等）
- **⏰ 时间序列最佳实践** - TimeSeriesSplit交叉验证，完全避免数据泄漏
- **🔍 智能超参优化** - Optuna贝叶斯优化，100次迭代TPE采样器
- **📈 完整指标支持** - 16个业务指标的模块化配置
- **🎯 生产级代码** - 模块化、测试、文档完备
- **📉 诊断工具** - 残差分析、Ljung-Box检验、误差分布
- **🚀 灵活推理** - 单点、批量、置信度估计

## 🏗️ 系统架构

### 级联融合流程

```
输入数据 → 特征工程(100+维) → CatBoost → 残差学习 → 权重融合 → 逆变换 → 预测
            (8个特征模块)   (第一层)      (第二层)   (α,β优化)
```

## 📁 项目结构

```
src/
├── data/             # 数据管道：加载、预处理、分割
├── features/         # 特征工程：8个特征模块，100+维特征
├── models/           # 模型层：CatBoost、LightGBM、级联融合
├── optimization/     # 超参优化：Optuna贝叶斯优化、权重学习
├── evaluation/       # 评估：指标、诊断、可视化
├── config/           # 配置管理：YAML配置加载
└── utils/            # 工具函数：日志、I/O、常量

config/
├── default.yaml      # 全局默认配置
├── metrics/          # 16个指标配置文件
└── features/         # 特征配置

examples/             # 可执行示例脚本
tests/                # 单元测试
docs/                 # 技术文档
```

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Alex-Wang66/GameRevenueForecast.git
cd GameRevenueForecast
pip install -r requirements.txt
```

### 基本使用

```python
from src.config.config import Config
from src.data.loader import DataLoader, DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.ensemble import CascadingEnsemble
from src.evaluation.metrics import evaluate_model

config = Config()
loader = DataLoader('data/raw/game_metrics.xlsx')
df = loader.load_excel()
df = DataPreprocessor.preprocess(df)

fe = FeatureEngineer(config)
X_train = fe.fit_transform(df, metric='income')

ensemble = CascadingEnsemble(config, use_learnable_weights=True)
ensemble.fit(X_train, y_train, X_val, y_val)
y_pred = ensemble.predict(X_test)

metrics = evaluate_model(y_test, y_pred)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")
```

## 📊 8个特征工程模块

| 模块 | 维度 | 说明 |
|------|------|------|
| temporal.py | 10 | 时间特征：小时、日期、月份、周等 |
| lag_features.py | 25 | 滞后特征：1,3,7,12,24小时 |
| rolling_stats.py | 20 | 滚动统计：sum/mean/std(3,7,12,24h) |
| cyclical.py | 10 | 循环编码：sin/cos变换 |
| holidays.py | 25 | 节假日：25个中国节假日+窗口 |
| interactions.py | 10 | 交互特征：特征组合 |
| transforms.py | - | 数据变换：标准化、对数变换 |
| engineer.py | 110+ | 编排器：统一管理 |

## 📈 支持的16个业务指标

income、paypeople、recall_income、income_after、income_paytime、
income_paytime_after、recall_income_paytime、recall_income_paytime_after、
payment_hour、paypeople_paytime、recall_paypeople_paytime、recall_paypeople、
regCount、recall_regCount、allLoginCount、allIncomeAfter

## 🎯 使用示例

```bash
# 训练单个指标
python examples/train_single_metric.py --metric income

# 批量训练16个指标
python examples/train_all_metrics.py

# 超参优化
python examples/hyperparameter_search.py --metric income --n_trials 100

# 完整工作流
python examples/train_ensemble.py --data_path data/raw/game_metrics.xlsx
```

## 🧪 运行测试

```bash
pytest tests/ -v
pytest --cov=src tests/
```

## 📚 文档

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构设计
- [FEATURES.md](docs/FEATURES.md) - 100+特征详解
- [MODELS.md](docs/MODELS.md) - 模型配置
- [OPTIMIZATION.md](docs/OPTIMIZATION.md) - 超参优化
- [WEIGHTS.md](docs/WEIGHTS.md) - 权重学习
- [API.md](docs/API.md) - API文档

## 🔨 技术栈

pandas, numpy, scikit-learn, CatBoost, LightGBM, Optuna, statsmodels, PyYAML, matplotlib, seaborn, pytest

## 📊 性能指标

| 指标 | Income | Paypeople | Recall_Income |
|------|--------|-----------|----------------|
| R² | 0.9494 | 0.9798 | 0.9552 |
| RMSE | 2310 | 755 | 1850 |
| SMAPE | 4.39% | 3.39% | 4.12% |

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 👤 作者

**Alex Wang** - [@Alex-Wang66](https://github.com/Alex-Wang66)

项目组合：GameRevenueForecast、SEC-HTML-Data-Miner、Basic-Learning-LangChain

## 📞 获取帮助

- 📖 查看 [文档](docs/)
- 🐛 提交 [Issue](https://github.com/Alex-Wang66/GameRevenueForecast/issues)
- 💬 参与 [讨论](https://github.com/Alex-Wang66/GameRevenueForecast/discussions)

---

**使用本项目进行预测时，请确保使用最新的训练模型权重。** 🚀
