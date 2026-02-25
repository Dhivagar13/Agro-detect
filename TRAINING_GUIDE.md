# AgroDetect AI - Complete Training Guide

## 📚 Step-by-Step Training Instructions

### Step 1: Download Dataset

#### Option A: PlantVillage Dataset (Recommended)

1. **Go to Kaggle:**
   - Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Or search "PlantVillage dataset" on Kaggle

2. **Download:**
   - Click "Download" button (requires Kaggle account)
   - File size: ~500MB
   - Contains 54,000+ images
   - 38 disease classes

3. **Extract:**
   - Extract the ZIP file to a folder
   - Example: `D:/datasets/plantvillage/`

#### Option B: Custom Dataset

Collect your own plant disease images and organize them.

### Step 2: Organize Dataset Structure

The dataset MUST be organized like this:

```
dataset/
├── Tomato___Early_blight/
│   ├── image001.jpg
│   ├── image002.jpg
│   ├── image003.jpg
│   └── ... (100+ images recommended)
│
├── Tomato___Late_blight/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
│
├── Tomato___healthy/
│   ├── image001.jpg
│   └── ...
│
├── Potato___Early_blight/
│   └── ...
│
├── Potato___Late_blight/
│   └── ...
│
└── ... (more disease classes)
```

**Important Rules:**
- Each disease = one folder
- Folder name = disease class name
- Use underscores (_) not spaces
- Format: `Crop___Disease` (3 underscores)
- Minimum 50-100 images per class
- Supported formats: JPG, JPEG, PNG, BMP

### Step 3: Place Dataset in Project

**Recommended Location:**

```
Agro-Detect/
├── data/
│   └── raw/
│       └── plantvillage/          ← Place dataset here
│           ├── Tomato___Early_blight/
│           ├── Tomato___Late_blight/
│           └── ...
```

**Or use any location:**
- `D:/datasets/plantvillage/`
- `C:/Users/YourName/Downloads/plantvillage/`
- Any path works!

### Step 4: Train the Model

#### Method 1: Using Command Line (Recommended)

1. **Open Terminal/PowerShell in project folder:**
   ```bash
   cd D:\My-Folder\Dhivagar-projects\Agro-Detect
   ```

2. **Activate virtual environment:**
   ```bash
   .\venv\Scripts\activate
   ```

3. **Run training command:**
   ```bash
   python train_model.py --data-dir "data/raw/plantvillage" --num-classes 38 --epochs 50
   ```

   **Or with full path:**
   ```bash
   python train_model.py --data-dir "D:/datasets/plantvillage" --num-classes 38 --epochs 50
   ```

#### Method 2: Quick Training Script

Create a file `quick_train.bat`:

```batch
@echo off
echo Starting AgroDetect AI Training...
echo.

cd /d D:\My-Folder\Dhivagar-projects\Agro-Detect
call venv\Scripts\activate

echo Training model on PlantVillage dataset...
python train_model.py --data-dir "data/raw/plantvillage" --num-classes 38 --epochs 50 --batch-size 32

echo.
echo Training complete!
pause
```

Then just double-click `quick_train.bat` to start training!

### Step 5: Training Parameters Explained

```bash
python train_model.py \
    --data-dir "path/to/dataset"    # Dataset location
    --num-classes 38                 # Number of disease classes
    --epochs 50                      # Training iterations (50-100 recommended)
    --batch-size 32                  # Images per batch (16-64)
    --learning-rate 0.001            # Learning speed (0.0001-0.01)
    --output-dir "models"            # Where to save model
```

**Parameter Guide:**

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--data-dir` | Required | Path to dataset folder |
| `--num-classes` | 38 (PlantVillage) | Number of disease types |
| `--epochs` | 50-100 | More = better accuracy (slower) |
| `--batch-size` | 32 | Higher = faster (needs more RAM) |
| `--learning-rate` | 0.001 | Default works well |

### Step 6: Monitor Training

During training, you'll see:

```
============================================================
AgroDetect AI - Model Training
============================================================

Found 38 classes: ['Apple___Apple_scab', 'Apple___Black_rot', ...]
Building model...
Model built successfully with 2592345 parameters
Preparing datasets...
Found 54305 images belonging to 38 classes
Training set: 43444 images
Validation set: 10861 images

Starting training...

Epoch 1/50
1358/1358 [==============================] - 180s 132ms/step
loss: 2.1234 - accuracy: 0.3250 - val_loss: 1.8765 - val_accuracy: 0.4500

Epoch 2/50
1358/1358 [==============================] - 175s 129ms/step
loss: 1.7654 - accuracy: 0.4875 - val_loss: 1.5432 - val_accuracy: 0.5750

...

Epoch 50/50
1358/1358 [==============================] - 172s 127ms/step
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

### Step 7: Training Time Estimates

**CPU Training:**
- Small dataset (5 classes, 1000 images): 30-60 minutes
- Medium dataset (15 classes, 5000 images): 1-2 hours
- Large dataset (38 classes, 54000 images): 3-5 hours

**GPU Training (if available):**
- Small: 5-10 minutes
- Medium: 15-30 minutes
- Large: 30-60 minutes

### Step 8: After Training

Once training completes, you'll have:

```
models/
├── plant_disease_model.h5          ← Trained model
├── class_names.json                ← Disease classes
├── training_history.json           ← Training metrics
└── checkpoints/
    ├── best_model.h5               ← Best performing model
    └── logs/                       ← TensorBoard logs
```

### Step 9: Use Trained Model

1. **Restart Streamlit:**
   ```bash
   streamlit run src/ui/app.py
   ```

2. **The app will automatically load the new trained model!**

3. **Upload plant images and get accurate predictions!**

## 🎯 Quick Start Example

### For PlantVillage Dataset:

```bash
# 1. Download dataset from Kaggle
# 2. Extract to D:/datasets/plantvillage/

# 3. Open PowerShell in project folder
cd D:\My-Folder\Dhivagar-projects\Agro-Detect

# 4. Activate environment
.\venv\Scripts\activate

# 5. Train model
python train_model.py --data-dir "D:/datasets/plantvillage" --num-classes 38 --epochs 50

# 6. Wait 3-5 hours for training to complete

# 7. Restart Streamlit
streamlit run src/ui/app.py

# 8. Upload images and get accurate predictions!
```

## 📊 Dataset Requirements

### Minimum Requirements:
- **Images per class:** 50-100 minimum
- **Image size:** Any size (will be resized to 224x224)
- **Image format:** JPG, JPEG, PNG, BMP
- **Total images:** 1000+ recommended

### Recommended:
- **Images per class:** 200-500
- **Image quality:** Clear, focused images
- **Variety:** Different angles, lighting, backgrounds
- **Balance:** Similar number of images per class

### Image Quality Tips:
✅ **Good Images:**
- Clear focus on leaf
- Good lighting
- Symptoms visible
- Minimal background
- Various angles

❌ **Poor Images:**
- Blurry
- Too dark/bright
- Multiple leaves overlapping
- Symptoms not visible
- Heavy filters

## 🔧 Troubleshooting

### Issue: "No module named 'tensorflow'"
**Solution:**
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution:**
```bash
# Reduce batch size
python train_model.py --data-dir "path" --num-classes 38 --batch-size 16
```

### Issue: "Not enough images"
**Solution:**
- Need minimum 50 images per class
- Download more images or use data augmentation

### Issue: Training is very slow
**Solution:**
- Reduce epochs: `--epochs 20`
- Reduce batch size: `--batch-size 16`
- Use fewer classes
- Consider GPU training

### Issue: Low accuracy after training
**Solution:**
- Train for more epochs: `--epochs 100`
- Check dataset quality
- Ensure balanced classes
- Verify image labels are correct

## 📈 Expected Results

### After Training:

**Accuracy:**
- Training: 90-96%
- Validation: 85-92%
- Real-world: 80-90%

**Confidence:**
- High confidence (>80%): Most predictions
- Medium (60-80%): Some predictions
- Low (<60%): Rare, usually poor image quality

**Speed:**
- Inference: 50-200ms per image
- Batch: 30-100ms per image

## 🎓 Advanced Training

### Fine-tuning:

After initial training, fine-tune for better accuracy:

```bash
# 1. Train with frozen base (fast)
python train_model.py --data-dir "path" --epochs 30

# 2. Fine-tune with unfrozen layers (slow but better)
# Edit train_model.py to unfreeze layers
python train_model.py --data-dir "path" --epochs 20 --learning-rate 0.0001
```

### Data Augmentation:

The training script automatically applies:
- Rotation (±20°)
- Horizontal/vertical flip
- Zoom (0.8-1.2x)
- Brightness adjustment (±20%)

### Transfer Learning:

The model uses MobileNetV2 pre-trained on ImageNet:
- Faster training
- Better accuracy with less data
- Efficient for mobile/edge devices

## 📞 Need Help?

Check these files:
- `AI_USAGE_GUIDE.md` - Detailed AI usage
- `EXAMPLES.md` - Code examples
- `PROJECT_STATUS.md` - Implementation status
- `logs/` - Training logs

---

**Ready to train?** Follow the steps above and you'll have an accurate plant disease detection model in a few hours! 🚀

**Version:** 1.0  
**Last Updated:** February 24, 2026
