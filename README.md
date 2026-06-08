# GameRevenueForecast 🎮📊

> 游戏变现分时预测系统 | Time-Series Forecasting for Game Monetization

## 📚 项目介绍

**GameRevenueForecast** 是一个专业的游戏变现分时预测系统，基于4399实习项目。采用**级联融合集成**和**可学习权重**等先进技术，能够精确预测游戏的16个关键商业指标。

### ✨ 核心特性

- **🤖 智能集成学习** - 两层级联融合 + 可学习权重
- **📊 100+维特征工程** - 滞后、滚动统计、循环编码、节假日影响
- **⏰ 时间序列最佳实践** - TimeSeriesSplit交叉验证
- **🔍 超参优化** - Optuna贝叶斯优化
- **📈 16个业务指标** - 完整支持所有关键变现指标
- **🎯 生产级质量** - 模块化、有测试、有文档

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

# 加载配置
config = Config()

# 加载和预处理数据
loader = DataLoader('data/raw/game_metrics.xlsx')
df = loader.load_excel()
df = DataPreprocessor.preprocess(df)
```

## 📖 文档

详见 `docs/` 目录的完整文档。

## 🔗 链接

- GitHub: https://github.com/Alex-Wang66/GameRevenueForecast
- 作者: [@Alex-Wang66](https://github.com/Alex-Wang66)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)
