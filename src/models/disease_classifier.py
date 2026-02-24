"""Disease Classifier using MobileNet transfer learning"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
from typing import Optional, List
from src.utils.logger import logger


class DiseaseClassifier:
    """
    Plant disease classifier using MobileNet transfer learning
    
    Leverages pre-trained MobileNetV2 weights from ImageNet and adds
    custom classification layers for disease-specific predictions.
    """
    
    def __init__(
        self,
        num_classes: int,
        input_shape: tuple = (224, 224, 3),
        base_model: str = 'mobilenet_v2'
    ):
        """
        Initialize Disease Classifier
        
        Args:
            num_classes: Number of disease classes to predict
            input_shape: Input image shape (height, width, channels)
            base_model: Base model architecture
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model_name = base_model
        self.model = None
        self.base_model = None
        
        logger.info(
            f"DiseaseClassifier initialized: {num_classes} classes, "
            f"input shape {input_shape}"
        )
    
    def build_model(self, weights: str = 'imagenet') -> keras.Model:
        """
        Build model with MobileNet base and custom classification head
        
        Args:
            weights: Pre-trained weights to load ('imagenet' or None)
        
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building model with {self.base_model_name} base")
        
        # Load MobileNetV2 base model
        self.base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights=weights
        )
        
        # Freeze base model layers initially
        self.base_model.trainable = False
        
        # Build custom classification head
        inputs = keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        logger.info(f"Model built successfully with {self.model.count_params()} parameters")
        return self.model
    
    def freeze_base_layers(self, num_layers: int = 100):
        """
        Freeze early layers of base model
        
        Args:
            num_layers: Number of layers to freeze from the start
        """
        if self.base_model is None:
            raise ValueError("Model must be built before freezing layers")
        
        self.base_model.trainable = True
        
        # Freeze first num_layers
        for layer in self.base_model.layers[:num_layers]:
            layer.trainable = False
        
        trainable_count = sum([1 for layer in self.base_model.layers if layer.trainable])
        logger.info(f"Frozen first {num_layers} layers, {trainable_count} layers trainable")
    
    def unfreeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Unfreeze specific layers for fine-tuning
        
        Args:
            layer_names: List of layer names to unfreeze (None = unfreeze all)
        """
        if self.base_model is None:
            raise ValueError("Model must be built before unfreezing layers")
        
        if layer_names is None:
            # Unfreeze all layers
            self.base_model.trainable = True
            logger.info("All base model layers unfrozen")
        else:
            # Unfreeze specific layers
            for layer in self.base_model.layers:
                if layer.name in layer_names:
                    layer.trainable = True
            logger.info(f"Unfrozen {len(layer_names)} specific layers")
    
    def compile_model(
        self,
        learning_rate: float = 0.001,
        optimizer: str = 'adam'
    ):
        """
        Compile model with optimizer and loss function
        
        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer name ('adam', 'sgd', etc.)
        """
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        if optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"Model compiled with {optimizer} optimizer, lr={learning_rate}")
    
    def save_model(self, save_path: str):
        """
        Save model weights and architecture
        
        Args:
            save_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(save_path))
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """
        Load model from checkpoint
        
        Args:
            model_path: Path to saved model
        """
        self.model = keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
