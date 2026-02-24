# AgroDetect AI - Usage Examples

## 🎯 Example 1: Using the Web Interface

### Step-by-Step

1. **Open the application**
   ```
   Browser → http://localhost:8501
   ```

2. **Navigate to Disease Classification**
   - Click "🔍 Disease Classification" in sidebar

3. **Upload an image**
   - Click "Browse files" or drag & drop
   - Select a leaf image (JPEG, PNG, or BMP)

4. **View results**
   - Disease name with confidence score
   - Alternative predictions
   - Treatment recommendations
   - Inference time

### Expected Output

```
✅ Analysis Complete!

Top Prediction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Disease Detected: Tomato - Early Blight
🟢 87.5% Confidence
Inference Time: 125.3 ms

Alternative Predictions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tomato - Early Blight     ████████████████████ 87.5%
Tomato - Late Blight      ███░░░░░░░░░░░░░░░░░  8.2%
Tomato - Septoria         ██░░░░░░░░░░░░░░░░░░  3.1%
Tomato - Healthy          █░░░░░░░░░░░░░░░░░░░  0.8%
Tomato - Leaf Mold        ░░░░░░░░░░░░░░░░░░░░  0.4%
```

## 🐍 Example 2: Python Script for Batch Processing

### Script: `batch_predict.py`

```python
"""Batch process multiple plant images"""

from pathlib import Path
import json
from src.inference.inference_engine import InferenceEngine

# Load model
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

engine = InferenceEngine(
    model_path='models/plant_disease_model.h5',
    class_names=class_names,
    confidence_threshold=0.7
)
engine.load_model()
engine.warm_up()

# Process images
image_dir = Path('data/test_images')
results = []

for image_path in image_dir.glob('*.jpg'):
    result = engine.predict_single(str(image_path))
    
    results.append({
        'image': image_path.name,
        'disease': result.disease_class,
        'confidence': result.confidence,
        'time_ms': result.inference_time_ms
    })
    
    print(f"{image_path.name}: {result.disease_class} ({result.confidence:.1f}%)")

# Save results
with open('predictions.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nProcessed {len(results)} images")
print(f"Results saved to predictions.json")
```

### Output

```
tomato_leaf_1.jpg: Tomato___Early_blight (87.5%)
tomato_leaf_2.jpg: Tomato___healthy (95.2%)
potato_leaf_1.jpg: Potato___Late_blight (82.3%)
corn_leaf_1.jpg: Corn_(maize)___Common_rust_ (78.9%)
grape_leaf_1.jpg: Grape___Black_rot (91.4%)

Processed 5 images
Results saved to predictions.json
```

## 🎓 Example 3: Training on Custom Dataset

### Dataset Structure

```
my_dataset/
├── Tomato___Early_blight/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ... (100+ images)
├── Tomato___Late_blight/
│   ├── img001.jpg
│   └── ... (100+ images)
├── Tomato___healthy/
│   └── ... (100+ images)
└── ... (more classes)
```

### Training Command

```bash
# Activate environment
.\venv\Scripts\activate

# Train model
python train_model.py \
    --data-dir "D:/datasets/my_dataset" \
    --num-classes 10 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --output-dir "models"
```

### Training Output

```
============================================================
AgroDetect AI - Model Training
============================================================

Found 10 classes: ['Tomato___Early_blight', 'Tomato___Late_blight', ...]
Building model...
Model built successfully with 2592345 parameters
Preparing datasets...
Found 1000 images belonging to 10 classes
Training set: 800 images
Validation set: 200 images

Starting training...

Epoch 1/50
25/25 [==============================] - 45s 1.8s/step
loss: 2.1234 - accuracy: 0.3250 - val_loss: 1.8765 - val_accuracy: 0.4500

Epoch 2/50
25/25 [==============================] - 42s 1.7s/step
loss: 1.7654 - accuracy: 0.4875 - val_loss: 1.5432 - val_accuracy: 0.5750

...

Epoch 50/50
25/25 [==============================] - 41s 1.6s/step
loss: 0.1234 - accuracy: 0.9625 - val_loss: 0.2345 - val_accuracy: 0.9250

============================================================
Training Complete!
Model saved to: models/plant_disease_model.h5
Class names saved to: models/class_names.json
Training history saved to: models/training_history.json
============================================================

Final Training Accuracy: 0.9625
Final Validation Accuracy: 0.9250
```

## 📊 Example 4: Analyzing Training Results

### Script: `analyze_training.py`

```python
"""Analyze training results"""

import json
import matplotlib.pyplot as plt

# Load training history
with open('models/training_history.json', 'r') as f:
    history = json.load(f)

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
print("Training analysis saved to training_results.png")

# Print summary
print("\nTraining Summary:")
print(f"Best Training Accuracy: {max(history['accuracy']):.4f}")
print(f"Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
print(f"Final Training Loss: {history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
```

## 🔬 Example 5: Testing Model Accuracy

### Script: `test_model.py`

```python
"""Test model on test dataset"""

from pathlib import Path
import json
from src.inference.inference_engine import InferenceEngine
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

engine = InferenceEngine(
    model_path='models/plant_disease_model.h5',
    class_names=class_names
)
engine.load_model()

# Test on dataset
test_dir = Path('data/test')
y_true = []
y_pred = []

for class_dir in test_dir.iterdir():
    if not class_dir.is_dir():
        continue
    
    true_label = class_dir.name
    
    for image_path in class_dir.glob('*.jpg'):
        result = engine.predict_single(str(image_path))
        
        y_true.append(true_label)
        y_pred.append(result.disease_class)

# Generate report
print("Classification Report:")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=class_names)
print("\nConfusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

### Output

```
Classification Report:
============================================================
                              precision    recall  f1-score   support

         Tomato___Early_blight       0.92      0.89      0.90        50
          Tomato___Late_blight       0.88      0.91      0.89        45
              Tomato___healthy       0.95      0.93      0.94        55
         Potato___Early_blight       0.87      0.85      0.86        40
          Potato___Late_blight       0.90      0.88      0.89        42
              Potato___healthy       0.93      0.95      0.94        48

                      accuracy                           0.91       280
                     macro avg       0.91      0.90      0.90       280
                  weighted avg       0.91      0.91      0.91       280

Overall Accuracy: 0.9107 (91.07%)
```

## 🌐 Example 6: REST API Integration (Future)

### Flask API Wrapper

```python
"""REST API for AgroDetect AI"""

from flask import Flask, request, jsonify
from src.inference.inference_engine import InferenceEngine
import json
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load model
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

engine = InferenceEngine(
    model_path='models/plant_disease_model.h5',
    class_names=class_names
)
engine.load_model()
engine.warm_up()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    image_array = np.array(image)
    
    # Get prediction
    result = engine.predict_single(image_array)
    
    return jsonify({
        'disease': result.disease_class,
        'confidence': result.confidence,
        'alternatives': result.probability_distribution,
        'inference_time_ms': result.inference_time_ms,
        'low_confidence': result.low_confidence_flag
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### API Usage

```bash
# Test health endpoint
curl http://localhost:5000/health

# Predict disease
curl -X POST \
  -F "image=@tomato_leaf.jpg" \
  http://localhost:5000/predict
```

### Response

```json
{
  "disease": "Tomato___Early_blight",
  "confidence": 87.5,
  "alternatives": {
    "Tomato___Early_blight": 87.5,
    "Tomato___Late_blight": 8.2,
    "Tomato___Septoria_leaf_spot": 3.1,
    "Tomato___healthy": 0.8,
    "Tomato___Leaf_Mold": 0.4
  },
  "inference_time_ms": 125.3,
  "low_confidence": false
}
```

## 📱 Example 7: Mobile Integration (Concept)

### React Native Component

```javascript
import React, { useState } from 'react';
import { View, Button, Image, Text } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

export default function PlantDiseaseDetector() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const pickImage = async () => {
    let result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });

    if (!result.cancelled) {
      setImage(result.uri);
      predictDisease(result.uri);
    }
  };

  const predictDisease = async (imageUri) => {
    const formData = new FormData();
    formData.append('image', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'leaf.jpg',
    });

    const response = await fetch('http://your-api.com/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setResult(data);
  };

  return (
    <View>
      <Button title="Take Photo" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={{ width: 300, height: 300 }} />}
      {result && (
        <View>
          <Text>Disease: {result.disease}</Text>
          <Text>Confidence: {result.confidence}%</Text>
        </View>
      )}
    </View>
  );
}
```

## 🎯 Example 8: Confidence Threshold Tuning

### Script: `tune_threshold.py`

```python
"""Find optimal confidence threshold"""

from pathlib import Path
import json
from src.inference.inference_engine import InferenceEngine
import numpy as np

# Load model
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Test different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
results = {}

for threshold in thresholds:
    engine = InferenceEngine(
        model_path='models/plant_disease_model.h5',
        class_names=class_names,
        confidence_threshold=threshold
    )
    engine.load_model()
    
    # Test on validation set
    test_dir = Path('data/validation')
    correct = 0
    total = 0
    flagged = 0
    
    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        for image_path in class_dir.glob('*.jpg'):
            result = engine.predict_single(str(image_path))
            
            if result.disease_class == class_dir.name:
                correct += 1
            
            if result.low_confidence_flag:
                flagged += 1
            
            total += 1
    
    accuracy = correct / total
    flag_rate = flagged / total
    
    results[threshold] = {
        'accuracy': accuracy,
        'flag_rate': flag_rate
    }
    
    print(f"Threshold {threshold}: Accuracy={accuracy:.3f}, Flag Rate={flag_rate:.3f}")

# Find optimal threshold
optimal = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nOptimal threshold: {optimal[0]} (Accuracy: {optimal[1]['accuracy']:.3f})")
```

---

**Examples Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Ready to Use
