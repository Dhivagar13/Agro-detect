"""Inference Engine for real-time disease prediction"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time

from src.preprocessing.image_preprocessor import ImagePreprocessor
from src.utils.logger import logger


@dataclass
class PredictionResult:
    """Result of disease prediction"""
    disease_class: str
    confidence: float  # 0-100%
    probability_distribution: Dict[str, float]
    inference_time_ms: float
    low_confidence_flag: bool


class InferenceEngine:
    """
    Inference engine for real-time disease prediction
    
    Loads optimized models and performs fast predictions with confidence scoring.
    """
    
    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        device: str = 'cpu',
        confidence_threshold: float = 0.7
    ):
        """
        Initialize Inference Engine
        
        Args:
            model_path: Path to trained model
            class_names: List of disease class names
            device: Device to use ('cpu' or 'gpu')
            confidence_threshold: Threshold for low confidence flagging
        """
        self.model_path = Path(model_path)
        self.class_names = class_names
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.preprocessor = ImagePreprocessor()
        
        logger.info(f"InferenceEngine initialized with device: {device}")
    
    def load_model(self):
        """Load TensorFlow or TFLite model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            if self.model_path.suffix == '.tflite':
                # Load TFLite model
                self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_tflite = True
                logger.info("TFLite model loaded successfully")
            else:
                # Load TensorFlow model
                self.model = keras.models.load_model(str(self.model_path))
                self.is_tflite = False
                logger.info("TensorFlow model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def warm_up(self, num_iterations: int = 10):
        """
        Warm up model to reduce first-inference latency
        
        Args:
            num_iterations: Number of warm-up iterations
        """
        if self.model is None and not hasattr(self, 'interpreter'):
            raise ValueError("Model must be loaded before warm-up")
        
        logger.info(f"Warming up model with {num_iterations} iterations")
        
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        for _ in range(num_iterations):
            if self.is_tflite:
                self.interpreter.set_tensor(self.input_details[0]['index'], dummy_input)
                self.interpreter.invoke()
            else:
                self.model.predict(dummy_input, verbose=0)
        
        logger.info("Model warm-up complete")
    
    def predict_single(
        self,
        image: Union[str, np.ndarray]
    ) -> PredictionResult:
        """
        Predict disease for a single image
        
        Args:
            image: Image file path or numpy array
        
        Returns:
            PredictionResult with disease classification and confidence
        """
        if self.model is None and not hasattr(self, 'interpreter'):
            raise ValueError("Model must be loaded before prediction")
        
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self.preprocessor.preprocess_single(image, normalize=True)
        preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
        
        # Run inference
        if self.is_tflite:
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            predictions = self.model.predict(preprocessed, verbose=0)[0]
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get confidence scores
        confidence_scores = self.get_confidence_scores(predictions)
        
        # Get top prediction
        top_class_idx = np.argmax(predictions)
        top_class = self.class_names[top_class_idx]
        top_confidence = float(predictions[top_class_idx] * 100)
        
        # Check if confidence is low
        low_confidence = top_confidence < (self.confidence_threshold * 100)
        
        result = PredictionResult(
            disease_class=top_class,
            confidence=top_confidence,
            probability_distribution=confidence_scores,
            inference_time_ms=inference_time,
            low_confidence_flag=low_confidence
        )
        
        logger.debug(
            f"Prediction: {top_class} ({top_confidence:.2f}%) "
            f"in {inference_time:.2f}ms"
        )
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]]
    ) -> List[PredictionResult]:
        """
        Predict diseases for a batch of images
        
        Args:
            images: List of image file paths or numpy arrays
        
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for image in images:
            try:
                result = self.predict_single(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict image: {str(e)}")
                continue
        
        logger.info(f"Batch prediction complete: {len(results)} images processed")
        return results
    
    def get_confidence_scores(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Extract confidence scores for all classes
        
        Args:
            predictions: Model prediction probabilities
        
        Returns:
            Dictionary mapping class names to confidence percentages
        """
        confidence_scores = {}
        
        for i, class_name in enumerate(self.class_names):
            confidence_scores[class_name] = float(predictions[i] * 100)
        
        # Sort by confidence (descending)
        confidence_scores = dict(
            sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return confidence_scores
