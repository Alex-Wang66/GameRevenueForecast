"""
Configuration management for GameRevenueForecast
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Central configuration manager"""

    def __init__(self, config_dir: str = None):
        """Initialize config manager"""
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), '../../config')

        self.config_dir = Path(config_dir)
        self.metrics_dir = self.config_dir / 'metrics'
        self.hyperparams_dir = self.config_dir / 'hyperparams'
        self.features_dir = self.config_dir / 'features'

        # Load default config
        self.default_config = self._load_yaml(self.config_dir / 'default.yaml')

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML config file"""
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}

        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def get_metric_config(self, metric_name: str) -> Dict[str, Any]:
        """Get configuration for specific metric"""
        config_path = self.metrics_dir / f'{metric_name}.yaml'
        return self._load_yaml(config_path)

    def get_hyperparams(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for model"""
        config_path = self.hyperparams_dir / f'{model_name}.yaml'
        return self._load_yaml(config_path)

    def list_metrics(self) -> list:
        """List all available metrics"""
        if not self.metrics_dir.exists():
            return []
        return [f.stem for f in self.metrics_dir.glob('*.yaml')]


class PathManager:
    """Manage project paths"""

    def __init__(self, project_root: str = None):
        """Initialize path manager"""
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.root = Path(project_root)
        self.src = self.root / 'src'
        self.data = self.root / 'data'
        self.config = self.root / 'config'
        self.models = self.data / 'models'
        self.results = self.data / 'results'

    def ensure_dirs_exist(self):
        """Create directories if they don't exist"""
        for dir_path in [self.data, self.models, self.results]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def get_raw_data_path(self) -> Path:
        """Get path to raw data"""
        return self.data / 'raw' / 'game_metrics.xlsx'

    def get_processed_data_path(self, name: str) -> Path:
        """Get path to processed data"""
        return self.data / 'processed' / f'{name}.parquet'

    def get_model_path(self, metric_name: str, layer: str = 'catboost') -> Path:
        """Get path to saved model"""
        return self.models / f'{layer}_{metric_name}.pkl'


# Create default instances
try:
    config = Config()
    paths = PathManager()
except Exception as e:
    logger.error(f"Error initializing config: {e}")
    config = None
    paths = None
