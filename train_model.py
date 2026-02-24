"""Train plant disease classification model"""

import argparse
from pathlib import Path
import json

from src.models.disease_classifier import DiseaseClassifier
from src.models.training_manager import TrainingManager, TrainingConfig
from src.utils.logger import logger


def train_model(
    data_dir: str,
    num_classes: int,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: str = 'models'
):
    """
    Train plant disease classification model
    
    Args:
        data_dir: Directory containing training data (organized by class folders)
        num_classes: Number of disease classes
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save trained model
    """
    
    logger.info("=" * 60)
    logger.info("AgroDetect AI - Model Training")
    logger.info("=" * 60)
    
    # Verify data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Get class names from directory structure
    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    logger.info(f"Found {len(class_names)} classes: {class_names[:5]}...")
    
    if len(class_names) != num_classes:
        logger.warning(f"Expected {num_classes} classes but found {len(class_names)}")
        num_classes = len(class_names)
    
    # Create classifier
    logger.info("Building model...")
    classifier = DiseaseClassifier(
        num_classes=num_classes,
        input_shape=(224, 224, 3)
    )
    classifier.build_model(weights='imagenet')
    classifier.compile_model(learning_rate=learning_rate)
    
    # Create training manager
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_split=0.2,
        early_stopping_patience=10,
        reduce_lr_patience=5
    )
    
    trainer = TrainingManager(classifier, config)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    train_ds, val_ds = trainer.prepare_dataset(
        data_dir=str(data_path),
        image_size=(224, 224),
        batch_size=batch_size
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_dataset=train_ds,
        val_dataset=val_ds,
        checkpoint_dir=f'{output_dir}/checkpoints'
    )
    
    # Save final model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / 'plant_disease_model.h5'
    classifier.save_model(str(model_path))
    
    # Save class names
    class_names_path = output_path / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    
    # Save training history
    history_path = output_path / 'training_history.json'
    trainer.save_training_history(str(history_path))
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Class names saved to: {class_names_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info("=" * 60)
    
    # Print final metrics
    final_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    logger.info(f"Final Training Accuracy: {final_acc:.4f}")
    logger.info(f"Final Validation Accuracy: {final_val_acc:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train plant disease classification model')
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing training data (organized by class folders)'
    )
    
    parser.add_argument(
        '--num-classes',
        type=int,
        required=True,
        help='Number of disease classes'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained model (default: models)'
    )
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
