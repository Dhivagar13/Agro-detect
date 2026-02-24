"""Dataset Manager for organizing and validating plant disease image datasets"""

import cv2
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from src.utils.logger import logger


@dataclass
class ValidationResult:
    """Result of image validation"""
    is_valid: bool
    image_path: str
    format: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    file_size_bytes: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class IngestResult:
    """Result of image ingestion process"""
    total_images: int
    valid_images: int
    invalid_images: int
    validation_results: List[ValidationResult]
    crop_type: str
    disease_class: str


class DatasetManager:
    """
    Manages plant disease image datasets including validation, organization, and versioning.
    
    Supports JPEG, PNG, and BMP image formats with validation for file integrity,
    format compliance, and basic quality checks.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize Dataset Manager
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.manifests_dir = self.base_dir / "manifests"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetManager initialized with base directory: {self.base_dir}")
    
    def validate_image(self, image_path: str) -> ValidationResult:
        """
        Validate an image file for format, integrity, and basic quality
        
        Args:
            image_path: Path to the image file
        
        Returns:
            ValidationResult with validation status and details
        """
        path = Path(image_path)
        
        # Check if file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                image_path=str(image_path),
                error_message="File does not exist"
            )
        
        # Check file extension
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return ValidationResult(
                is_valid=False,
                image_path=str(image_path),
                error_message=f"Unsupported format. Supported: {self.SUPPORTED_FORMATS}"
            )
        
        # Get file size
        file_size = path.stat().st_size
        
        # Try to read image with OpenCV
        try:
            image = cv2.imread(str(path))
            
            if image is None:
                return ValidationResult(
                    is_valid=False,
                    image_path=str(image_path),
                    format=path.suffix.lower(),
                    file_size_bytes=file_size,
                    error_message="Failed to read image - file may be corrupted"
                )
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Check minimum dimensions (at least 10x10 pixels)
            if height < 10 or width < 10:
                return ValidationResult(
                    is_valid=False,
                    image_path=str(image_path),
                    format=path.suffix.lower(),
                    dimensions=(width, height),
                    file_size_bytes=file_size,
                    error_message=f"Image too small: {width}x{height} (minimum 10x10)"
                )
            
            # Validation successful
            return ValidationResult(
                is_valid=True,
                image_path=str(image_path),
                format=path.suffix.lower(),
                dimensions=(width, height),
                file_size_bytes=file_size
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                image_path=str(image_path),
                format=path.suffix.lower(),
                file_size_bytes=file_size,
                error_message=f"Error reading image: {str(e)}"
            )
    
    def ingest_images(
        self,
        directory_path: str,
        crop_type: str,
        disease_class: str
    ) -> IngestResult:
        """
        Ingest images from a directory, validate them, and prepare for organization
        
        Args:
            directory_path: Path to directory containing images
            crop_type: Type of crop (e.g., 'tomato', 'potato')
            disease_class: Disease classification (e.g., 'early_blight', 'healthy')
        
        Returns:
            IngestResult with statistics and validation results
        """
        source_dir = Path(directory_path)
        
        if not source_dir.exists():
            logger.error(f"Source directory does not exist: {directory_path}")
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        logger.info(f"Ingesting images from {directory_path} for {crop_type}/{disease_class}")
        
        # Find all image files
        image_files = []
        for ext in self.SUPPORTED_FORMATS:
            image_files.extend(source_dir.glob(f"*{ext}"))
            image_files.extend(source_dir.glob(f"*{ext.upper()}"))
        
        # Validate each image
        validation_results = []
        valid_count = 0
        invalid_count = 0
        
        for image_file in image_files:
            result = self.validate_image(str(image_file))
            validation_results.append(result)
            
            if result.is_valid:
                valid_count += 1
                logger.debug(f"Valid image: {image_file.name}")
            else:
                invalid_count += 1
                logger.warning(f"Invalid image: {image_file.name} - {result.error_message}")
        
        ingest_result = IngestResult(
            total_images=len(image_files),
            valid_images=valid_count,
            invalid_images=invalid_count,
            validation_results=validation_results,
            crop_type=crop_type,
            disease_class=disease_class
        )
        
        logger.info(
            f"Ingestion complete: {valid_count} valid, {invalid_count} invalid "
            f"out of {len(image_files)} total images"
        )
        
        return ingest_result
    
    def organize_dataset(
        self,
        source_dir: str,
        target_dir: Optional[str] = None,
        crop_type: str = None,
        disease_class: str = None,
        copy_files: bool = True
    ) -> Dict[str, any]:
        """
        Organize images into hierarchical directory structure by crop type and disease class
        
        Args:
            source_dir: Source directory containing images
            target_dir: Target directory for organized dataset (default: processed_dir)
            crop_type: Crop type for organization
            disease_class: Disease class for organization
            copy_files: If True, copy files; if False, move files
        
        Returns:
            Dictionary with organization statistics
        """
        if target_dir is None:
            target_dir = self.processed_dir
        else:
            target_dir = Path(target_dir)
        
        # Create target directory structure
        organized_dir = target_dir / crop_type / disease_class
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Organizing dataset: {source_dir} -> {organized_dir}")
        
        # Ingest and validate images
        ingest_result = self.ingest_images(source_dir, crop_type, disease_class)
        
        # Copy/move valid images to organized structure
        organized_count = 0
        failed_count = 0
        
        for validation_result in ingest_result.validation_results:
            if validation_result.is_valid:
                source_path = Path(validation_result.image_path)
                target_path = organized_dir / source_path.name
                
                try:
                    if copy_files:
                        shutil.copy2(source_path, target_path)
                    else:
                        shutil.move(str(source_path), str(target_path))
                    organized_count += 1
                except Exception as e:
                    logger.error(f"Failed to organize {source_path.name}: {str(e)}")
                    failed_count += 1
        
        stats = {
            'total_images': ingest_result.total_images,
            'valid_images': ingest_result.valid_images,
            'organized_images': organized_count,
            'failed_images': failed_count,
            'target_directory': str(organized_dir)
        }
        
        logger.info(f"Organization complete: {organized_count} images organized")
        return stats
    
    def save_metadata(
        self,
        image_path: str,
        crop_type: str,
        disease_class: str,
        source: str = "user_upload",
        additional_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Save metadata for an image
        
        Args:
            image_path: Path to the image
            crop_type: Crop type
            disease_class: Disease classification
            source: Source of the image
            additional_metadata: Additional metadata to store
        
        Returns:
            Complete metadata dictionary
        """
        path = Path(image_path)
        
        # Validate image to get dimensions and format
        validation_result = self.validate_image(image_path)
        
        metadata = {
            'image_id': path.stem,
            'filename': path.name,
            'upload_timestamp': datetime.now().isoformat(),
            'file_size_bytes': validation_result.file_size_bytes,
            'dimensions': {
                'width': validation_result.dimensions[0] if validation_result.dimensions else None,
                'height': validation_result.dimensions[1] if validation_result.dimensions else None
            },
            'format': validation_result.format,
            'crop_type': crop_type,
            'disease_label': disease_class,
            'source': source,
            'storage_path': str(path)
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Save metadata to JSON file
        metadata_path = path.parent / f"{path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Metadata saved for {path.name}")
        return metadata
    
    def get_dataset_stats(self, dataset_path: str) -> Dict:
        """
        Calculate statistics for a dataset
        
        Args:
            dataset_path: Path to the dataset directory
        
        Returns:
            Dictionary with dataset statistics
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        logger.info(f"Calculating statistics for dataset: {dataset_path}")
        
        # Initialize statistics
        stats = {
            'total_images': 0,
            'crop_types': {},
            'disease_classes': {},
            'class_distribution': {},
            'total_size_bytes': 0,
            'average_dimensions': {'width': 0, 'height': 0}
        }
        
        total_width = 0
        total_height = 0
        
        # Walk through directory structure
        for crop_dir in dataset_dir.iterdir():
            if not crop_dir.is_dir():
                continue
            
            crop_type = crop_dir.name
            stats['crop_types'][crop_type] = 0
            
            for disease_dir in crop_dir.iterdir():
                if not disease_dir.is_dir():
                    continue
                
                disease_class = disease_dir.name
                class_key = f"{crop_type}/{disease_class}"
                
                # Count images in this class
                image_count = 0
                for ext in self.SUPPORTED_FORMATS:
                    image_count += len(list(disease_dir.glob(f"*{ext}")))
                    image_count += len(list(disease_dir.glob(f"*{ext.upper()}")))
                
                stats['class_distribution'][class_key] = image_count
                stats['crop_types'][crop_type] += image_count
                stats['disease_classes'][disease_class] = stats['disease_classes'].get(disease_class, 0) + image_count
                stats['total_images'] += image_count
                
                # Calculate size and dimensions
                for image_file in disease_dir.iterdir():
                    if image_file.suffix.lower() in self.SUPPORTED_FORMATS:
                        stats['total_size_bytes'] += image_file.stat().st_size
                        
                        # Get dimensions
                        try:
                            img = cv2.imread(str(image_file))
                            if img is not None:
                                h, w = img.shape[:2]
                                total_width += w
                                total_height += h
                        except:
                            pass
        
        # Calculate averages
        if stats['total_images'] > 0:
            stats['average_dimensions']['width'] = total_width // stats['total_images']
            stats['average_dimensions']['height'] = total_height // stats['total_images']
            stats['average_size_mb'] = stats['total_size_bytes'] / (1024 * 1024 * stats['total_images'])
        
        logger.info(f"Dataset statistics: {stats['total_images']} images across {len(stats['crop_types'])} crop types")
        return stats
    
    def create_version(
        self,
        dataset_path: str,
        version_tag: str,
        description: Optional[str] = None
    ) -> Dict:
        """
        Create a versioned snapshot of a dataset
        
        Args:
            dataset_path: Path to the dataset directory
            version_tag: Version identifier (e.g., 'v1.0', '2024-01-15')
            description: Optional description of this version
        
        Returns:
            Dictionary with version information
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        logger.info(f"Creating dataset version: {version_tag}")
        
        # Get dataset statistics
        stats = self.get_dataset_stats(dataset_path)
        
        # Create version metadata
        version_info = {
            'version_tag': version_tag,
            'creation_date': datetime.now().isoformat(),
            'dataset_path': str(dataset_dir),
            'description': description,
            'statistics': stats
        }
        
        # Save version info
        version_dir = self.manifests_dir / version_tag
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = version_dir / 'version_info.json'
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Dataset version {version_tag} created successfully")
        return version_info
    
    def generate_manifest(
        self,
        dataset_path: str,
        version_tag: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate a manifest file listing all images and their labels
        
        Args:
            dataset_path: Path to the dataset directory
            version_tag: Version identifier
            output_path: Optional custom output path for manifest
        
        Returns:
            Dictionary containing the manifest data
        """
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        logger.info(f"Generating manifest for dataset version: {version_tag}")
        
        # Initialize manifest
        manifest = {
            'version_tag': version_tag,
            'generation_date': datetime.now().isoformat(),
            'dataset_path': str(dataset_dir),
            'images': []
        }
        
        # Walk through directory structure and collect image information
        for crop_dir in dataset_dir.iterdir():
            if not crop_dir.is_dir():
                continue
            
            crop_type = crop_dir.name
            
            for disease_dir in crop_dir.iterdir():
                if not disease_dir.is_dir():
                    continue
                
                disease_class = disease_dir.name
                
                # Find all images
                for ext in self.SUPPORTED_FORMATS:
                    for image_file in disease_dir.glob(f"*{ext}"):
                        # Validate image
                        validation_result = self.validate_image(str(image_file))
                        
                        if validation_result.is_valid:
                            image_entry = {
                                'filename': image_file.name,
                                'relative_path': str(image_file.relative_to(dataset_dir)),
                                'crop_type': crop_type,
                                'disease_class': disease_class,
                                'format': validation_result.format,
                                'dimensions': {
                                    'width': validation_result.dimensions[0],
                                    'height': validation_result.dimensions[1]
                                },
                                'file_size_bytes': validation_result.file_size_bytes
                            }
                            manifest['images'].append(image_entry)
        
        # Add summary statistics
        manifest['summary'] = {
            'total_images': len(manifest['images']),
            'crop_types': list(set(img['crop_type'] for img in manifest['images'])),
            'disease_classes': list(set(img['disease_class'] for img in manifest['images']))
        }
        
        # Determine output path
        if output_path is None:
            version_dir = self.manifests_dir / version_tag
            version_dir.mkdir(parents=True, exist_ok=True)
            output_path = version_dir / 'manifest.json'
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(
            f"Manifest generated: {len(manifest['images'])} images, "
            f"saved to {output_path}"
        )
        
        return manifest
