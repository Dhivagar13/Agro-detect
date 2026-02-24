"""Model modules for AgroDetect AI"""

from .disease_classifier import DiseaseClassifier
from .training_manager import TrainingManager, TrainingConfig

__all__ = ['DiseaseClassifier', 'TrainingManager', 'TrainingConfig']
