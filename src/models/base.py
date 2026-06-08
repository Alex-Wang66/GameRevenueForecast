"""
Base model interface
"""

from abc import ABC, abstractmethod
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base interface for all models"""

    def __init__(self, model_name: str = None):
        """Initialize base model"""
        self.model_name = model_name or self.__class__.__name__
        self.is_fitted = False
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit model on training data"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        pass

    def fit_predict(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Fit and predict in one step"""
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def get_feature_importance(self, feature_names: list = None):
        """Get feature importance if available"""
        if not hasattr(self.model, 'get_feature_importance'):
            logger.warning(f"{self.model_name} does not support feature importance")
            return None
        
        return self.model.get_feature_importance(feature_names=feature_names)
