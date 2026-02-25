# Dataset Setup Guide - Visual Instructions

## 📁 Where to Place Your Dataset

### Option 1: Inside Project Folder (Recommended)

```
D:\My-Folder\Dhivagar-projects\Agro-Detect\
│
├── data/
│   └── raw/
│       └── plantvillage/          ← PUT DATASET HERE
│           ├── Apple___Apple_scab/
│           │   ├── image1.jpg
│           │   ├── image2.jpg
│           │   └── ...
│           ├── Apple___Black_rot/
│           ├── Tomato___Early_blight/
│           └── ...
│
├── models/                        ← Trained model saves here
├── src/
├── venv/
└── train_model.py
```

**Command to use:**
```bash
python train_model.py --data-dir "data/raw/plantvillage" --num-classes 38 --epochs 50
```

### Option 2: Separate Location

```
D:\datasets\
└── plantvillage/                  ← PUT DATASET HERE
    ├── Apple___Apple_scab/
    ├── Apple___Black_rot/
    ├── Tomato___Early_blight/
    └── ...
```

**Command to use:**
```bash
python train_model.py --data-dir "D:/datasets/plantvillage" --num-classes 38 --epochs 50
```

## 📥 Step-by-Step Setup

### Step 1: Download Dataset

1. **Go to Kaggle:**
   - Open browser
   - Visit: https://www.kaggle.com/datasets/emmarex/plantdisease
   - Login or create account (free)

2. **Download:**
   - Click blue "Download" button
   - File: `archive.zip` (~500MB)
   - Save to Downloads folder

### Step 2: Extract Dataset

1. **Find downloaded file:**
   - Go to Downloads folder
   - Find `archive.zip`

2. **Extract:**
   - Right-click on `archive.zip`
   - Select "Extract All..."
   - Choose destination:
     - Option A: `D:\My-Folder\Dhivagar-projects\Agro-Detect\data\raw\`
     - Option B: `D:\datasets\`
   - Click "Extract"

3. **Verify structure:**
   - Open extracted folder
   - Should see folders like:
     - `Apple___Apple_scab`
     - `Apple___Black_rot`
     - `Tomato___Early_blight`
     - etc.

### Step 3: Verify Dataset Structure

**Correct Structure:**
```
plantvillage/
├── Apple___Apple_scab/
│   ├── 00a822cf-2a1a-4f79-8c3f-2c7d8e5e5e5e___FREC_Scab 3417.JPG
│   ├── 00b1f5e5-3c4d-4e5f-8c3f-2c7d8e5e5e5e___FREC_Scab 3418.JPG
│   └── ... (more images)
├── Apple___Black_rot/
│   └── ... (images)
├── Tomato___Early_blight/
│   └── ... (images)
└── ... (38 folders total)
```

**Each folder should contain:**
- 100-2000 images
- JPG or PNG format
- Images of diseased/healthy leaves

## 🚀 Quick Training Methods

### Method 1: Double-Click Batch File (Easiest!)

1. **Find file:** `quick_train.bat` in project folder
2. **Double-click** to run
3. **Enter information:**
   - Dataset path: `data/raw/plantvillage` or `D:/datasets/plantvillage`
   - Number of classes: `38`
   - Epochs: `50`
4. **Press Enter** and wait!

### Method 2: Command Line

1. **Open PowerShell:**
   - Press `Win + X`
   - Select "Windows PowerShell"

2. **Navigate to project:**
   ```bash
   cd D:\My-Folder\Dhivagar-projects\Agro-Detect
   ```

3. **Activate environment:**
   ```bash
   .\venv\Scripts\activate
   ```

4. **Run training:**
   ```bash
   python train_model.py --data-dir "data/raw/plantvillage" --num-classes 38 --epochs 50
   ```

### Method 3: Python Script

Create `start_training.py`:

```python
import os
import subprocess

# Configuration
DATASET_PATH = "data/raw/plantvillage"  # Change this to your path
NUM_CLASSES = 38
EPOCHS = 50
BATCH_SIZE = 32

# Run training
command = f'python train_model.py --data-dir "{DATASET_PATH}" --num-classes {NUM_CLASSES} --epochs {EPOCHS} --batch-size {BATCH_SIZE}'

print("Starting training...")
print(f"Dataset: {DATASET_PATH}")
print(f"Classes: {NUM_CLASSES}")
print(f"Epochs: {EPOCHS}")
print()

subprocess.run(command, shell=True)
```

Then run:
```bash
python start_training.py
```

## 📊 Dataset Information

### PlantVillage Dataset

**Total Images:** 54,305  
**Classes:** 38  
**Crops:** 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)

**Class Distribution:**

| Crop | Diseases | Images |
|------|----------|--------|
| Tomato | 10 classes | ~18,000 |
| Potato | 3 classes | ~3,000 |
| Corn | 4 classes | ~4,000 |
| Grape | 4 classes | ~5,000 |
| Apple | 4 classes | ~3,000 |
| Others | 13 classes | ~21,000 |

### Custom Dataset

If using your own images:

**Requirements:**
- Minimum 50 images per disease
- Clear, focused images
- Consistent lighting
- Organized in folders by disease

**Folder Naming:**
- Format: `Crop___Disease`
- Use 3 underscores (___) between crop and disease
- Examples:
  - `Tomato___Early_blight`
  - `Potato___Late_blight`
  - `Corn___Common_rust`
  - `Tomato___healthy`

## ⏱️ Training Time

### Expected Duration:

**CPU (Intel i5/i7):**
- 5 classes, 1000 images: 30-60 minutes
- 15 classes, 5000 images: 1-2 hours
- 38 classes, 54000 images: 3-5 hours

**GPU (NVIDIA GTX/RTX):**
- 5 classes: 5-10 minutes
- 15 classes: 15-30 minutes
- 38 classes: 30-60 minutes

### Progress Indicators:

During training you'll see:
```
Epoch 1/50
1358/1358 [==============================] - 180s 132ms/step
loss: 2.1234 - accuracy: 0.3250 - val_loss: 1.8765 - val_accuracy: 0.4500
```

- `Epoch X/50`: Current progress
- `accuracy`: Training accuracy (should increase)
- `val_accuracy`: Validation accuracy (target: >85%)
- `180s`: Time per epoch

## ✅ Verification Checklist

Before training, verify:

- [ ] Dataset downloaded and extracted
- [ ] Folder structure is correct (crop___disease format)
- [ ] Each folder contains images
- [ ] Virtual environment is activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Enough disk space (2GB+ free)
- [ ] Enough RAM (8GB+ recommended)

## 🎯 After Training

### Files Created:

```
models/
├── plant_disease_model.h5          ← Main model (14MB)
├── class_names.json                ← Disease names
├── training_history.json           ← Metrics
└── checkpoints/
    ├── best_model.h5               ← Best version
    └── logs/                       ← TensorBoard logs
```

### Next Steps:

1. **Restart Streamlit:**
   ```bash
   streamlit run src/ui/app.py
   ```

2. **Test the model:**
   - Go to "AI Scanner" page
   - Upload a plant leaf image
   - Get accurate prediction!

3. **Check accuracy:**
   - Should see high confidence (>80%)
   - Correct disease identification
   - Fast inference (<200ms)

## 🆘 Common Issues

### Issue: "Dataset not found"
**Solution:** Check path is correct
```bash
# Use forward slashes or double backslashes
python train_model.py --data-dir "D:/datasets/plantvillage"
# OR
python train_model.py --data-dir "D:\\datasets\\plantvillage"
```

### Issue: "Not enough images"
**Solution:** Need minimum 50 images per class

### Issue: "Out of memory"
**Solution:** Reduce batch size
```bash
python train_model.py --data-dir "path" --num-classes 38 --batch-size 16
```

### Issue: "Python not found"
**Solution:** Activate virtual environment first
```bash
.\venv\Scripts\activate
```

## 📞 Need Help?

1. Check `TRAINING_GUIDE.md` for detailed instructions
2. Check `AI_USAGE_GUIDE.md` for usage examples
3. Check `logs/` folder for error messages
4. Verify dataset structure matches examples above

---

**Ready to train?** Follow the steps above and you'll have an accurate model in a few hours! 🚀
