"""Download pre-trained plant disease model"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import json

from src.models.disease_classifier import DiseaseClassifier
from src.utils.logger import logger


# PlantVillage dataset classes
PLANT_DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


def create_and_train_simple_model():
    """Create and train a simple model for demonstration"""
    
    logger.info("Creating plant disease classifier")
    
    # Create classifier
    classifier = DiseaseClassifier(
        num_classes=len(PLANT_DISEASE_CLASSES),
        input_shape=(224, 224, 3)
    )
    
    # Build model
    classifier.build_model(weights='imagenet')
    classifier.compile_model(learning_rate=0.001)
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / 'plant_disease_model.h5'
    classifier.save_model(str(model_path))
    
    # Save class names
    class_names_path = model_dir / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(PLANT_DISEASE_CLASSES, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Class names saved to {class_names_path}")
    
    return model_path, class_names_path


if __name__ == '__main__':
    print("=" * 60)
    print("AgroDetect AI - Model Setup")
    print("=" * 60)
    print()
    print("Creating pre-trained model architecture...")
    print(f"Number of disease classes: {len(PLANT_DISEASE_CLASSES)}")
    print()
    
    model_path, class_names_path = create_and_train_simple_model()
    
    print()
    print("✓ Model setup complete!")
    print()
    print("Note: This model uses MobileNetV2 with ImageNet weights.")
    print("For accurate disease detection, you need to train it on")
    print("plant disease images using the TrainingManager.")
    print()
    print("To train the model:")
    print("1. Organize your dataset in folders by disease class")
    print("2. Use the training script or Streamlit UI to train")
    print("3. The model will learn to detect specific plant diseases")
    print()
    print("=" * 60)
