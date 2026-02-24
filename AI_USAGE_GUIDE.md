# AgroDetect AI - Usage Guide

## 🎯 Overview

AgroDetect AI uses deep learning (MobileNetV2 with transfer learning) to analyze plant leaf images and detect diseases. The system can identify 25+ different plant diseases across multiple crops including tomatoes, potatoes, corn, grapes, and apples.

## 🚀 Quick Start

### 1. Setup (Already Done!)

The system is already set up with:
- ✅ Python environment with all dependencies
- ✅ Pre-trained MobileNetV2 model architecture
- ✅ Streamlit web interface
- ✅ Inference engine for predictions

### 2. Using the Web Interface

The Streamlit app is running at: **http://localhost:8501**

**To analyze a plant disease:**

1. Navigate to "🔍 Disease Classification" page
2. Upload a clear image of a plant leaf (JPEG, PNG, or BMP)
3. Wait for AI analysis (takes 1-2 seconds)
4. View results:
   - Top disease prediction with confidence score
   - Alternative predictions
   - Disease information and treatment recommendations

**Confidence Indicators:**
- 🟢 Green (80%+): High confidence - reliable prediction
- 🟡 Yellow (60-80%): Medium confidence - likely correct
- 🔴 Red (<60%): Low confidence - retake image with better lighting

## 🧠 How the AI Works

### Model Architecture

```
Input Image (224x224x3)
    ↓
MobileNetV2 Base (Pre-trained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense Layer (256 neurons, ReLU)
    ↓
Dropout (50%)
    ↓
Output Layer (25 classes, Softmax)
```

### Transfer Learning

The model uses **transfer learning**:
1. **Base Model**: MobileNetV2 trained on ImageNet (1.4M images, 1000 classes)
2. **Feature Extraction**: Leverages learned features (edges, textures, patterns)
3. **Custom Head**: New layers trained specifically for plant diseases

### Current Model Status

⚠️ **Important**: The current model has the architecture but needs training on plant disease images for accurate predictions.

**What it can do now:**
- Accept and process leaf images
- Run inference through the neural network
- Provide predictions with confidence scores

**What it needs:**
- Training on labeled plant disease dataset
- Fine-tuning for specific disease patterns

## 📚 Training Your Own Model

### Option 1: Use Existing Dataset

Download a plant disease dataset like PlantVillage:
- **PlantVillage Dataset**: 54,000+ images of healthy and diseased plant leaves
- **Download**: https://www.kaggle.com/datasets/emmarex/plantdisease

### Option 2: Collect Your Own Data

Organize images in this structure:
```
dataset/
├── Tomato___Early_blight/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Tomato___Late_blight/
│   ├── image1.jpg
│   └── ...
├── Tomato___healthy/
│   └── ...
└── ...
```

### Training Command

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Train model
python train_model.py \
    --data-dir "path/to/dataset" \
    --num-classes 25 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

**Training Parameters:**
- `--data-dir`: Path to organized dataset
- `--num-classes`: Number of disease classes
- `--epochs`: Training iterations (50-100 recommended)
- `--batch-size`: Images per batch (32 for most GPUs)
- `--learning-rate`: Learning rate (0.001 is good default)

**Training Time:**
- CPU: 2-4 hours for 50 epochs
- GPU: 20-40 minutes for 50 epochs

### After Training

The trained model will be saved to:
- `models/plant_disease_model.h5` - Trained model
- `models/class_names.json` - Disease class names
- `models/training_history.json` - Training metrics

Restart the Streamlit app to use the new model!

## 🎯 Best Practices for Accurate Predictions

### Image Quality

**Good Images:**
- ✅ Clear, focused leaf image
- ✅ Good lighting (natural daylight preferred)
- ✅ Leaf fills most of the frame
- ✅ Symptoms clearly visible
- ✅ Minimal background clutter

**Poor Images:**
- ❌ Blurry or out of focus
- ❌ Too dark or overexposed
- ❌ Leaf too small in frame
- ❌ Multiple leaves overlapping
- ❌ Heavy shadows

### When to Trust Predictions

**High Confidence (80%+):**
- Prediction is likely accurate
- Proceed with recommended treatment
- Monitor plant for changes

**Medium Confidence (60-80%):**
- Prediction is probably correct
- Consider consulting an expert
- Take additional photos from different angles

**Low Confidence (<60%):**
- Retake image with better quality
- Try different lighting conditions
- Consult agricultural expert

## 🔧 Advanced Features

### Batch Processing

Process multiple images programmatically:

```python
from src.inference.inference_engine import InferenceEngine
import json

# Load model
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

engine = InferenceEngine(
    model_path='models/plant_disease_model.h5',
    class_names=class_names
)
engine.load_model()

# Process multiple images
image_paths = ['leaf1.jpg', 'leaf2.jpg', 'leaf3.jpg']
results = engine.predict_batch(image_paths)

for img, result in zip(image_paths, results):
    print(f"{img}: {result.disease_class} ({result.confidence:.1f}%)")
```

### Model Optimization

For edge devices (Raspberry Pi, mobile):

```python
from src.models.disease_classifier import DiseaseClassifier

# Load trained model
classifier = DiseaseClassifier(num_classes=25)
classifier.load_model('models/plant_disease_model.h5')

# Convert to TFLite (smaller, faster)
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(classifier.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open('models/plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## 📊 Supported Diseases

Current model supports 25 disease classes:

**Apple:**
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

**Corn:**
- Cercospora Leaf Spot
- Common Rust
- Northern Leaf Blight
- Healthy

**Grape:**
- Black Rot
- Esca (Black Measles)
- Leaf Blight
- Healthy

**Potato:**
- Early Blight
- Late Blight
- Healthy

**Tomato:**
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Yellow Leaf Curl Virus
- Mosaic Virus
- Healthy

## 🆘 Troubleshooting

### Model Not Loading

**Error**: "Model not loaded" in sidebar

**Solution**:
```bash
python download_pretrained_model.py
```

### Low Accuracy Predictions

**Causes**:
- Model not trained on plant disease data
- Poor image quality
- Disease not in training set

**Solutions**:
- Train model on labeled dataset
- Improve image quality
- Add more disease classes

### Slow Predictions

**Causes**:
- Large model size
- CPU inference
- No model warm-up

**Solutions**:
- Use GPU if available
- Convert to TFLite
- Reduce image size

## 📈 Next Steps

1. **Get Training Data**: Download PlantVillage or collect your own images
2. **Train Model**: Run training script with your dataset
3. **Evaluate**: Test on validation images
4. **Deploy**: Use trained model in Streamlit app
5. **Monitor**: Track prediction accuracy and user feedback

## 🤝 Support

For issues or questions:
- Check logs in `logs/` directory
- Review training history in `models/training_history.json`
- Consult PROJECT_STATUS.md for implementation details

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Model**: MobileNetV2 Transfer Learning
