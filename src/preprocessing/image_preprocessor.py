"""Image Preprocessor for preparing images for MobileNet model"""

import cv2
import numpy as np
from typing import Tuple, List, Union, Optional
from pathlib import Path
from src.utils.logger import logger


class ImagePreprocessor:
    """
    Preprocesses images for MobileNet model input
    
    Handles resizing, normalization, and color space conversion to prepare
    images for inference or training with MobileNet architecture.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize Image Preprocessor
        
        Args:
            target_size: Target dimensions (width, height) for resized images
        """
        self.target_size = target_size
        logger.info(f"ImagePreprocessor initialized with target size: {target_size}")
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Resize image to target dimensions
        
        Args:
            image: Input image as numpy array
            target_size: Optional override for target size
        
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        # OpenCV uses (width, height) format
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [-1, 1] range for MobileNet
        
        MobileNet expects pixel values in the range [-1, 1]
        
        Args:
            image: Input image with pixel values in [0, 255]
        
        Returns:
            Normalized image with pixel values in [-1, 1]
        """
        # Convert to float32
        normalized = image.astype(np.float32)
        
        # Scale from [0, 255] to [-1, 1]
        normalized = (normalized / 127.5) - 1.0
        
        return normalized
    
    def convert_color_space(
        self,
        image: np.ndarray,
        target_space: str = 'RGB'
    ) -> np.ndarray:
        """
        Convert image color space
        
        Args:
            image: Input image
            target_space: Target color space ('RGB', 'BGR', 'GRAY')
        
        Returns:
            Converted image
        """
        # Detect current color space based on shape
        if len(image.shape) == 2:
            # Grayscale image
            if target_space == 'RGB':
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif target_space == 'BGR':
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                return image
        
        elif len(image.shape) == 3:
            channels = image.shape[2]
            
            if channels == 4:
                # RGBA/BGRA - remove alpha channel first
                image = image[:, :, :3]
            
            # Assume BGR (OpenCV default) and convert if needed
            if target_space == 'RGB':
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif target_space == 'BGR':
                return image
            elif target_space == 'GRAY':
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        return image
    
    def preprocess_single(
        self,
        image_input: Union[str, np.ndarray],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess a single image for model input
        
        Args:
            image_input: Image file path or numpy array
            normalize: Whether to normalize pixel values
        
        Returns:
            Preprocessed image ready for model input
        """
        # Load image if path is provided
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Failed to load image: {image_input}")
        else:
            image = image_input.copy()
        
        # Convert to RGB (MobileNet expects RGB)
        image = self.convert_color_space(image, target_space='RGB')
        
        # Resize to target dimensions
        image = self.resize_image(image)
        
        # Normalize if requested
        if normalize:
            image = self.normalize_pixels(image)
        
        return image
    
    def preprocess_batch(
        self,
        image_inputs: List[Union[str, np.ndarray]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess a batch of images
        
        Args:
            image_inputs: List of image file paths or numpy arrays
            normalize: Whether to normalize pixel values
        
        Returns:
            Batch of preprocessed images as numpy array with shape (batch_size, height, width, channels)
        """
        preprocessed_images = []
        
        for image_input in image_inputs:
            try:
                preprocessed = self.preprocess_single(image_input, normalize=normalize)
                preprocessed_images.append(preprocessed)
            except Exception as e:
                logger.error(f"Failed to preprocess image: {str(e)}")
                continue
        
        if not preprocessed_images:
            raise ValueError("No images were successfully preprocessed")
        
        # Stack into batch
        batch = np.stack(preprocessed_images, axis=0)
        
        logger.debug(f"Preprocessed batch of {len(preprocessed_images)} images")
        return batch
    
    def preprocess_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for display (denormalize if needed)
        
        Args:
            image: Preprocessed image (possibly normalized)
        
        Returns:
            Image suitable for display with pixel values in [0, 255]
        """
        # Check if image is normalized (values in [-1, 1])
        if image.min() < 0 or image.max() <= 1.0:
            # Denormalize from [-1, 1] to [0, 255]
            image = ((image + 1.0) * 127.5).astype(np.uint8)
        elif image.max() <= 1.0:
            # Scale from [0, 1] to [0, 255]
            image = (image * 255).astype(np.uint8)
        else:
            # Already in [0, 255] range
            image = image.astype(np.uint8)
        
        return image
