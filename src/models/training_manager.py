"""Training Manager for model training and evaluation"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import json

from src.models.disease_classifier import DiseaseClassifier
from src.utils.logger import logger


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    save_best_only: bool = True


class TrainingManager:
    """
    Manages model training, evaluation, and checkpointing
    """
    
    def __init__(
        self,
        classifier: DiseaseClassifier,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize Training Manager
        
        Args:
            classifier: DiseaseClassifier instance
            config: Training configuration
        """
        self.classifier = classifier
        self.config = config or TrainingConfig()
        self.history = None
        
        logger.info("TrainingManager initialized")
    
    def prepare_dataset(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training and validation datasets
        
        Args:
            data_dir: Directory containing class subdirectories
            image_size: Target image size
            batch_size: Batch size (uses config if None)
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        batch_size = batch_size or self.config.batch_size
        
        logger.info(f"Loading dataset from {data_dir}")
        
        # Load training dataset
        train_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config.validation_split,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Load validation dataset
        val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=self.config.validation_split,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size
        )
        
        # Normalize pixel values
        normalization_layer = keras.layers.Rescaling(1./127.5, offset=-1)
        train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
        
        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        logger.info("Dataset prepared successfully")
        return train_ds, val_ds
    
    def get_callbacks(self, checkpoint_dir: str) -> list:
        """
        Create training callbacks
        
        Args:
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            List of Keras callbacks
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        callbacks = [
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.config.reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=self.config.save_best_only,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir=str(checkpoint_path / 'logs'),
                histogram_freq=1
            )
        ]
        
        logger.info(f"Callbacks configured: {len(callbacks)} callbacks")
        return callbacks
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        checkpoint_dir: str = 'models/checkpoints'
    ) -> Dict:
        """
        Train the model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Training history dictionary
        """
        if self.classifier.model is None:
            raise ValueError("Model must be built before training")
        
        logger.info("Starting model training")
        
        callbacks = self.get_callbacks(checkpoint_dir)
        
        self.history = self.classifier.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training complete")
        return self.history.history
    
    def evaluate(
        self,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.classifier.model is None:
            raise ValueError("Model must be built before evaluation")
        
        logger.info("Evaluating model")
        
        results = self.classifier.model.evaluate(test_dataset, verbose=1)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                  (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def save_training_history(self, save_path: str):
        """
        Save training history to JSON
        
        Args:
            save_path: Path to save history
        """
        if self.history is None:
            raise ValueError("No training history to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.history.history, f, indent=2)
        
        logger.info(f"Training history saved to {save_path}")
