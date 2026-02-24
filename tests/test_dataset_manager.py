"""Tests for DatasetManager"""

import pytest
import cv2
import numpy as np
from hypothesis import given, strategies as st, settings
from pathlib import Path
import tempfile
import shutil

from src.data.dataset_manager import DatasetManager, ValidationResult, IngestResult


class TestDatasetManagerUnit:
    """Unit tests for DatasetManager"""
    
    @pytest.fixture
    def dataset_manager(self, tmp_path):
        """Create a DatasetManager instance with temporary directory"""
        return DatasetManager(base_dir=str(tmp_path / "data"))
    
    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image"""
        def _create_image(format='jpg', width=100, height=100, valid=True):
            image_path = tmp_path / f"test_image.{format}"
            
            if valid:
                # Create a valid image
                img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                cv2.imwrite(str(image_path), img)
            else:
                # Create an invalid/corrupted file
                with open(image_path, 'wb') as f:
                    f.write(b'invalid image data')
            
            return str(image_path)
        
        return _create_image
    
    def test_validate_image_valid_jpeg(self, dataset_manager, temp_image):
        """Test validation of valid JPEG image"""
        image_path = temp_image(format='jpg', valid=True)
        result = dataset_manager.validate_image(image_path)
        
        assert result.is_valid
        assert result.format == '.jpg'
        assert result.dimensions == (100, 100)
        assert result.error_message is None
    
    def test_validate_image_valid_png(self, dataset_manager, temp_image):
        """Test validation of valid PNG image"""
        image_path = temp_image(format='png', valid=True)
        result = dataset_manager.validate_image(image_path)
        
        assert result.is_valid
        assert result.format == '.png'
    
    def test_validate_image_valid_bmp(self, dataset_manager, temp_image):
        """Test validation of valid BMP image"""
        image_path = temp_image(format='bmp', valid=True)
        result = dataset_manager.validate_image(image_path)
        
        assert result.is_valid
        assert result.format == '.bmp'
    
    def test_validate_image_corrupted(self, dataset_manager, temp_image):
        """Test validation of corrupted image"""
        image_path = temp_image(format='jpg', valid=False)
        result = dataset_manager.validate_image(image_path)
        
        assert not result.is_valid
        assert "corrupted" in result.error_message.lower() or "failed" in result.error_message.lower()
    
    def test_validate_image_nonexistent(self, dataset_manager):
        """Test validation of non-existent file"""
        result = dataset_manager.validate_image("nonexistent.jpg")
        
        assert not result.is_valid
        assert "does not exist" in result.error_message.lower()
    
    def test_validate_image_unsupported_format(self, dataset_manager, tmp_path):
        """Test validation of unsupported format"""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")
        
        result = dataset_manager.validate_image(str(text_file))
        
        assert not result.is_valid
        assert "unsupported" in result.error_message.lower()
    
    def test_validate_image_too_small(self, dataset_manager, temp_image):
        """Test validation of image with dimensions too small"""
        image_path = temp_image(format='jpg', width=5, height=5, valid=True)
        result = dataset_manager.validate_image(image_path)
        
        assert not result.is_valid
        assert "too small" in result.error_message.lower()
    
    def test_ingest_images_valid_directory(self, dataset_manager, tmp_path):
        """Test ingesting images from a directory"""
        # Create test images
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(test_dir / f"image_{i}.jpg"), img)
        
        result = dataset_manager.ingest_images(
            str(test_dir),
            crop_type="tomato",
            disease_class="healthy"
        )
        
        assert result.total_images == 3
        assert result.valid_images == 3
        assert result.invalid_images == 0
        assert result.crop_type == "tomato"
        assert result.disease_class == "healthy"
    
    def test_ingest_images_mixed_validity(self, dataset_manager, tmp_path):
        """Test ingesting directory with both valid and invalid images"""
        test_dir = tmp_path / "test_images"
        test_dir.mkdir()
        
        # Create valid images
        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(test_dir / f"valid_{i}.jpg"), img)
        
        # Create invalid file
        (test_dir / "invalid.jpg").write_bytes(b'corrupted')
        
        result = dataset_manager.ingest_images(
            str(test_dir),
            crop_type="potato",
            disease_class="blight"
        )
        
        assert result.total_images == 3
        assert result.valid_images == 2
        assert result.invalid_images == 1
    
    def test_organize_dataset(self, dataset_manager, tmp_path):
        """Test organizing dataset into hierarchical structure"""
        # Create source directory with images
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(source_dir / f"image_{i}.jpg"), img)
        
        # Organize dataset
        stats = dataset_manager.organize_dataset(
            str(source_dir),
            crop_type="tomato",
            disease_class="early_blight"
        )
        
        assert stats['valid_images'] == 3
        assert stats['organized_images'] == 3
        
        # Check directory structure was created
        target_dir = Path(stats['target_directory'])
        assert target_dir.exists()
        assert target_dir.name == "early_blight"
        assert target_dir.parent.name == "tomato"
    
    def test_save_metadata(self, dataset_manager, temp_image):
        """Test saving metadata for an image"""
        image_path = temp_image(format='jpg', valid=True)
        
        metadata = dataset_manager.save_metadata(
            image_path,
            crop_type="potato",
            disease_class="healthy",
            source="test_upload"
        )
        
        assert metadata['crop_type'] == "potato"
        assert metadata['disease_label'] == "healthy"
        assert metadata['source'] == "test_upload"
        assert metadata['format'] == '.jpg'
        assert 'upload_timestamp' in metadata
        
        # Check metadata file was created
        metadata_file = Path(image_path).parent / f"{Path(image_path).stem}_metadata.json"
        assert metadata_file.exists()
    
    def test_get_dataset_stats(self, dataset_manager, tmp_path):
        """Test calculating dataset statistics"""
        # Create organized dataset structure
        dataset_dir = tmp_path / "dataset"
        
        # Create tomato/healthy
        tomato_healthy = dataset_dir / "tomato" / "healthy"
        tomato_healthy.mkdir(parents=True)
        for i in range(5):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tomato_healthy / f"image_{i}.jpg"), img)
        
        # Create tomato/blight
        tomato_blight = dataset_dir / "tomato" / "blight"
        tomato_blight.mkdir(parents=True)
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tomato_blight / f"image_{i}.jpg"), img)
        
        stats = dataset_manager.get_dataset_stats(str(dataset_dir))
        
        assert stats['total_images'] == 8
        assert stats['crop_types']['tomato'] == 8
        assert stats['class_distribution']['tomato/healthy'] == 5
        assert stats['class_distribution']['tomato/blight'] == 3
    
    def test_create_version(self, dataset_manager, tmp_path):
        """Test creating a dataset version"""
        # Create a simple dataset
        dataset_dir = tmp_path / "dataset"
        crop_dir = dataset_dir / "tomato" / "healthy"
        crop_dir.mkdir(parents=True)
        
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(crop_dir / "image.jpg"), img)
        
        version_info = dataset_manager.create_version(
            str(dataset_dir),
            version_tag="v1.0",
            description="Initial version"
        )
        
        assert version_info['version_tag'] == "v1.0"
        assert version_info['description'] == "Initial version"
        assert 'creation_date' in version_info
        assert 'statistics' in version_info
    
    def test_generate_manifest(self, dataset_manager, tmp_path):
        """Test generating a manifest file"""
        # Create a simple dataset
        dataset_dir = tmp_path / "dataset"
        crop_dir = dataset_dir / "tomato" / "healthy"
        crop_dir.mkdir(parents=True)
        
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(crop_dir / f"image_{i}.jpg"), img)
        
        manifest = dataset_manager.generate_manifest(
            str(dataset_dir),
            version_tag="v1.0"
        )
        
        assert manifest['version_tag'] == "v1.0"
        assert len(manifest['images']) == 3
        assert manifest['summary']['total_images'] == 3
        assert 'tomato' in manifest['summary']['crop_types']
        assert 'healthy' in manifest['summary']['disease_classes']
        
        # Check each image entry
        for img_entry in manifest['images']:
            assert 'filename' in img_entry
            assert 'crop_type' in img_entry
            assert 'disease_class' in img_entry
            assert img_entry['crop_type'] == 'tomato'
            assert img_entry['disease_class'] == 'healthy'


class TestDatasetManagerProperties:
    """Property-based tests for DatasetManager"""
    
    @pytest.fixture
    def dataset_manager(self, tmp_path):
        """Create a DatasetManager instance with temporary directory"""
        return DatasetManager(base_dir=str(tmp_path / "data"))
    
    @pytest.mark.property
    @given(
        width=st.integers(min_value=10, max_value=1000),
        height=st.integers(min_value=10, max_value=1000),
        format=st.sampled_from(['jpg', 'jpeg', 'png', 'bmp'])
    )
    @settings(max_examples=20, deadline=None)
    def test_property_multi_format_support(self, dataset_manager, tmp_path, width, height, format):
        """
        Property 1: Multi-format image support
        For any valid image file in JPEG, PNG, or BMP format, 
        the Dataset_Manager should successfully ingest and validate the image
        
        Validates: Requirements 1.1, 1.2
        """
        # Create a valid image in the specified format
        image_path = tmp_path / f"test_image.{format}"
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)
        
        # Validate the image
        result = dataset_manager.validate_image(str(image_path))
        
        # Property: All supported formats should be validated successfully
        assert result.is_valid, f"Failed to validate {format} image: {result.error_message}"
        assert result.format.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
        assert result.dimensions == (width, height)
        assert result.file_size_bytes > 0
        assert result.error_message is None

    @pytest.mark.property
    @given(
        crop_type=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu'))),
        disease_class=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Ll', 'Lu'))),
        num_images=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=15, deadline=None)
    def test_property_directory_organization_consistency(self, dataset_manager, tmp_path, crop_type, disease_class, num_images):
        """
        Property 2: Directory organization consistency
        For any set of images with crop type and disease class metadata,
        the Dataset_Manager should organize them into the hierarchical structure
        {crop_type}/{disease_class}/ and all images should be findable at their expected paths
        
        Validates: Requirements 1.3
        """
        # Create source directory with images
        source_dir = tmp_path / "source"
        source_dir.mkdir()
        
        created_files = []
        for i in range(num_images):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            image_path = source_dir / f"image_{i}.jpg"
            cv2.imwrite(str(image_path), img)
            created_files.append(f"image_{i}.jpg")
        
        # Organize dataset
        stats = dataset_manager.organize_dataset(
            str(source_dir),
            crop_type=crop_type,
            disease_class=disease_class
        )
        
        # Property: All valid images should be organized into correct directory structure
        target_dir = Path(stats['target_directory'])
        assert target_dir.exists()
        assert target_dir.name == disease_class
        assert target_dir.parent.name == crop_type
        
        # Property: All images should be findable at expected paths
        organized_files = list(target_dir.glob("*.jpg"))
        assert len(organized_files) == num_images
        
        for filename in created_files:
            expected_path = target_dir / filename
            assert expected_path.exists(), f"Image {filename} not found at expected path"
    
    @pytest.mark.property
    @given(
        crop_type=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('Ll',))),
        disease_class=st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=('Ll',))),
        source=st.sampled_from(['user_upload', 'training_dataset', 'validation'])
    )
    @settings(max_examples=15, deadline=None)
    def test_property_metadata_preservation(self, dataset_manager, tmp_path, crop_type, disease_class, source):
        """
        Property 3: Metadata preservation
        For any image with associated metadata (source, capture date, disease labels),
        after ingestion and organization, all metadata should be preserved and retrievable
        
        Validates: Requirements 1.4
        """
        # Create a test image
        image_path = tmp_path / "test_image.jpg"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(image_path), img)
        
        # Save metadata with additional fields
        additional_metadata = {
            'capture_date': '2024-01-15',
            'location': 'test_farm',
            'notes': 'test image'
        }
        
        metadata = dataset_manager.save_metadata(
            str(image_path),
            crop_type=crop_type,
            disease_class=disease_class,
            source=source,
            additional_metadata=additional_metadata
        )
        
        # Property: All metadata should be preserved
        assert metadata['crop_type'] == crop_type
        assert metadata['disease_label'] == disease_class
        assert metadata['source'] == source
        assert metadata['capture_date'] == '2024-01-15'
        assert metadata['location'] == 'test_farm'
        assert metadata['notes'] == 'test image'
        assert 'upload_timestamp' in metadata
        assert 'file_size_bytes' in metadata
        assert 'dimensions' in metadata
        
        # Property: Metadata should be retrievable from file
        import json
        metadata_file = Path(image_path).parent / f"{Path(image_path).stem}_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata['crop_type'] == crop_type
        assert loaded_metadata['disease_label'] == disease_class
        assert loaded_metadata['source'] == source

    @pytest.mark.property
    @given(
        num_crops=st.integers(min_value=1, max_value=3),
        images_per_class=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=10, deadline=None)
    def test_property_manifest_completeness(self, dataset_manager, tmp_path, num_crops, images_per_class):
        """
        Property 4: Manifest completeness
        For any dataset version, the generated manifest should contain entries
        for all images in the dataset with their correct labels
        
        Validates: Requirements 1.6
        """
        # Create a dataset with multiple crops and classes
        dataset_dir = tmp_path / "dataset"
        
        expected_images = []
        crop_names = [f"crop{i}" for i in range(num_crops)]
        disease_names = ["healthy", "diseased"]
        
        for crop in crop_names:
            for disease in disease_names:
                class_dir = dataset_dir / crop / disease
                class_dir.mkdir(parents=True)
                
                for i in range(images_per_class):
                    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
                    image_name = f"image_{i}.jpg"
                    cv2.imwrite(str(class_dir / image_name), img)
                    expected_images.append({
                        'crop': crop,
                        'disease': disease,
                        'filename': image_name
                    })
        
        # Generate manifest
        manifest = dataset_manager.generate_manifest(
            str(dataset_dir),
            version_tag="test_version"
        )
        
        # Property: Manifest should contain all images
        assert len(manifest['images']) == len(expected_images)
        assert manifest['summary']['total_images'] == len(expected_images)
        
        # Property: Each image should have correct labels
        manifest_images = {
            (img['crop_type'], img['disease_class'], img['filename']): img
            for img in manifest['images']
        }
        
        for expected in expected_images:
            key = (expected['crop'], expected['disease'], expected['filename'])
            assert key in manifest_images, f"Image {expected['filename']} not found in manifest"
            
            manifest_entry = manifest_images[key]
            assert manifest_entry['crop_type'] == expected['crop']
            assert manifest_entry['disease_class'] == expected['disease']
            assert manifest_entry['filename'] == expected['filename']
            assert 'dimensions' in manifest_entry
            assert 'file_size_bytes' in manifest_entry
            assert 'format' in manifest_entry
        
        # Property: Summary should accurately reflect dataset composition
        assert set(manifest['summary']['crop_types']) == set(crop_names)
        assert set(manifest['summary']['disease_classes']) == set(disease_names)
