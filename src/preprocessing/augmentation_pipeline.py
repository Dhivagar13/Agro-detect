"""Augmentation Pipeline for generating diverse training samples"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random
from src.utils.logger import logger


@dataclass
class AugmentationConfig:
    """Configuration for image augmentation"""
    rotation_range: int = 20  # degrees
    horizontal_flip: bool = True
    vertical_flip: bool = True
    zoom_range: Tuple[float, float] = (0.8, 1.2)
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    apply_noise: bool = False
    noise_std: float = 0.01


class AugmentationPipeline:
    """
    Generates augmented training samples through image transformations
    
    Applies random combinations of rotation, flipping, zooming, and brightness
    adjustments while preserving disease-relevant features.
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        """
        Initialize Augmentation Pipeline
        
        Args:
            config: Augmentation configuration
        """
        self.config = config if config is not None else AugmentationConfig()
        logger.info(f"AugmentationPipeline initialized with config: {self.config}")
    
    def rotate_image(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by specified angle
        
        Args:
            image: Input image
            angle: Rotation angle in degrees (random if None)
        
        Returns:
            Rotated image
        """
        if angle is None:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return rotated
    
    def flip_image(
        self,
        image: np.ndarray,
        horizontal: Optional[bool] = None,
        vertical: Optional[bool] = None
    ) -> np.ndarray:
        """
        Flip image horizontally and/or vertically
        
        Args:
            image: Input image
            horizontal: Flip horizontally (random if None)
            vertical: Flip vertically (random if None)
        
        Returns:
            Flipped image
        """
        result = image.copy()
        
        # Horizontal flip
        if horizontal is None:
            horizontal = self.config.horizontal_flip and random.random() > 0.5
        if horizontal:
            result = cv2.flip(result, 1)
        
        # Vertical flip
        if vertical is None:
            vertical = self.config.vertical_flip and random.random() > 0.5
        if vertical:
            result = cv2.flip(result, 0)
        
        return result
    
    def zoom_image(self, image: np.ndarray, zoom_factor: Optional[float] = None) -> np.ndarray:
        """
        Zoom in/out on image
        
        Args:
            image: Input image
            zoom_factor: Zoom factor (random if None)
        
        Returns:
            Zoomed image
        """
        if zoom_factor is None:
            zoom_factor = random.uniform(self.config.zoom_range[0], self.config.zoom_range[1])
        
        height, width = image.shape[:2]
        
        # Calculate new dimensions
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Crop or pad to original size
        if zoom_factor > 1.0:
            # Crop center
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            result = resized[start_y:start_y + height, start_x:start_x + width]
        else:
            # Pad to original size
            result = np.zeros_like(image)
            start_y = (height - new_height) // 2
            start_x = (width - new_width) // 2
            result[start_y:start_y + new_height, start_x:start_x + new_width] = resized
        
        return result
    
    def adjust_brightness(
        self,
        image: np.ndarray,
        brightness_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input image
            brightness_factor: Brightness multiplier (random if None)
        
        Returns:
            Brightness-adjusted image
        """
        if brightness_factor is None:
            brightness_factor = random.uniform(
                self.config.brightness_range[0],
                self.config.brightness_range[1]
            )
        
        # Apply brightness adjustment
        adjusted = image.astype(np.float32) * brightness_factor
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to image
        
        Args:
            image: Input image
        
        Returns:
            Noisy image
        """
        if not self.config.apply_noise:
            return image
        
        noise = np.random.normal(0, self.config.noise_std * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def augment_image(
        self,
        image: np.ndarray,
        num_augmentations: int = 1
    ) -> List[np.ndarray]:
        """
        Apply random combination of augmentations to image
        
        Args:
            image: Input image
            num_augmentations: Number of augmented versions to generate
        
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for _ in range(num_augmentations):
            result = image.copy()
            
            # Randomly apply transformations
            if random.random() > 0.5:
                result = self.rotate_image(result)
            
            if random.random() > 0.5:
                result = self.flip_image(result)
            
            if random.random() > 0.5:
                result = self.zoom_image(result)
            
            if random.random() > 0.5:
                result = self.adjust_brightness(result)
            
            if self.config.apply_noise and random.random() > 0.5:
                result = self.add_noise(result)
            
            augmented_images.append(result)
        
        return augmented_images
    
    def augment_batch(
        self,
        images: List[np.ndarray],
        augmentations_per_image: int = 1
    ) -> np.ndarray:
        """
        Augment a batch of images
        
        Args:
            images: List of input images
            augmentations_per_image: Number of augmented versions per image
        
        Returns:
            Batch of augmented images as numpy array
        """
        all_augmented = []
        
        for image in images:
            augmented = self.augment_image(image, num_augmentations=augmentations_per_image)
            all_augmented.extend(augmented)
        
        # Stack into batch
        batch = np.stack(all_augmented, axis=0)
        
        logger.debug(
            f"Augmented batch: {len(images)} images -> {len(all_augmented)} augmented images"
        )
        
        return batch
    
    def configure_transforms(
        self,
        rotation_range: Optional[int] = None,
        horizontal_flip: Optional[bool] = None,
        vertical_flip: Optional[bool] = None,
        zoom_range: Optional[Tuple[float, float]] = None,
        brightness_range: Optional[Tuple[float, float]] = None
    ):
        """
        Update augmentation configuration
        
        Args:
            rotation_range: Rotation angle range in degrees
            horizontal_flip: Enable horizontal flipping
            vertical_flip: Enable vertical flipping
            zoom_range: Zoom factor range
            brightness_range: Brightness multiplier range
        """
        if rotation_range is not None:
            self.config.rotation_range = rotation_range
        if horizontal_flip is not None:
            self.config.horizontal_flip = horizontal_flip
        if vertical_flip is not None:
            self.config.vertical_flip = vertical_flip
        if zoom_range is not None:
            self.config.zoom_range = zoom_range
        if brightness_range is not None:
            self.config.brightness_range = brightness_range
        
        logger.info(f"Augmentation configuration updated: {self.config}")
    
    def get_augmentation_stats(self) -> dict:
        """
        Get current augmentation configuration as dictionary
        
        Returns:
            Dictionary with augmentation settings
        """
        return {
            'rotation_range': self.config.rotation_range,
            'horizontal_flip': self.config.horizontal_flip,
            'vertical_flip': self.config.vertical_flip,
            'zoom_range': self.config.zoom_range,
            'brightness_range': self.config.brightness_range,
            'apply_noise': self.config.apply_noise,
            'noise_std': self.config.noise_std
        }
