# AgroDetect AI

An intelligent plant disease classification system using transfer learning with MobileNet CNN architecture.

## Features

- 🌱 Real-time plant disease detection from leaf images
- 🚀 Transfer learning with MobileNet for efficient training
- 📊 Interactive Streamlit interface for classification and analytics
- 🔒 Secure authentication and role-based access control
- 📈 Comprehensive analytics dashboard with visualizations
- 🎯 Edge-optimized models for deployment on resource-constrained devices

## Project Structure

```
agrodetect-ai/
├── src/                    # Source code
│   ├── data/              # Data management modules
│   │   └── dataset_manager.py
│   ├── preprocessing/     # Image preprocessing
│   │   ├── image_preprocessor.py
│   │   └── augmentation_pipeline.py
│   ├── models/            # Model architecture and training
│   │   ├── disease_classifier.py
│   │   ├── training_manager.py
│   │   └── model_optimizer.py
│   ├── inference/         # Inference engine
│   │   └── inference_engine.py
│   ├── auth/              # Authentication services
│   ├── ui/                # Streamlit interface
│   │   └── app.py
│   └── utils/             # Utilities
│       └── logger.py
├── tests/                 # Test suite
│   └── test_dataset_manager.py
├── data/                  # Data storage
│   ├── raw/              # Raw images
│   ├── processed/        # Processed datasets
│   └── manifests/        # Dataset manifests
├── models/                # Trained models
├── config/                # Configuration files
│   ├── __init__.py
│   └── config.yaml
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── setup.bat             # Windows setup script
├── setup.sh              # Linux/Mac setup script
└── README.md             # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

#### Windows

1. Run the setup script:
```bash
setup.bat
```

2. Activate the virtual environment:
```bash
venv\Scripts\activate
```

#### Linux/Mac

1. Make the setup script executable and run it:
```bash
chmod +x setup.sh
./setup.sh
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

#### Manual Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run src/ui/app.py
```

The app will open in your default browser at `http://localhost:8501`

### Training a Model

```python
from src.models.disease_classifier import DiseaseClassifier
from src.models.training_manager import TrainingManager
from src.data.dataset_manager import DatasetManager

# Prepare dataset
dataset_manager = DatasetManager()
dataset_manager.organize_dataset(
    source_dir="path/to/images",
    crop_type="tomato",
    disease_class="early_blight"
)

# Initialize classifier
classifier = DiseaseClassifier(num_classes=10)
classifier.build_model()
classifier.compile_model()

# Train model (requires TensorFlow dataset)
# trainer = TrainingManager(classifier)
# trainer.train(train_dataset, val_dataset)
```

### Making Predictions

```python
from src.inference.inference_engine import InferenceEngine

# Initialize inference engine
engine = InferenceEngine(
    model_path="models/disease_classifier.h5",
    class_names=["healthy", "early_blight", "late_blight"]
)
engine.load_model()

# Predict single image
result = engine.predict_single("path/to/leaf_image.jpg")
print(f"Disease: {result.disease_class}")
print(f"Confidence: {result.confidence:.2f}%")
```

### Managing Datasets

```python
from src.data.dataset_manager import DatasetManager

# Initialize dataset manager
dm = DatasetManager()

# Ingest and validate images
result = dm.ingest_images(
    directory_path="path/to/images",
    crop_type="tomato",
    disease_class="healthy"
)

print(f"Valid images: {result.valid_images}")
print(f"Invalid images: {result.invalid_images}")

# Create dataset version
version_info = dm.create_version(
    dataset_path="data/processed",
    version_tag="v1.0",
    description="Initial dataset"
)

# Generate manifest
manifest = dm.generate_manifest(
    dataset_path="data/processed",
    version_tag="v1.0"
)
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m property      # Property-based tests only
pytest -m integration   # Integration tests only

# Run with coverage
pytest --cov=src --cov-report=html

# View coverage report
# Open htmlcov/index.html in your browser
```

## Configuration

Edit `config/config.yaml` to customize:

- Model architecture and hyperparameters
- Training configuration
- Augmentation settings
- Inference parameters
- Logging settings
- Streamlit UI settings

## Development

### Code Style

This project follows PEP 8 style guidelines. Format code using:

```bash
black src/ tests/
flake8 src/ tests/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement feature with tests
3. Run tests: `pytest`
4. Commit changes: `git commit -m "Add your feature"`
5. Push branch: `git push origin feature/your-feature`

### Testing Strategy

- **Unit tests**: Test individual components in isolation
- **Property-based tests**: Test invariants using Hypothesis
- **Integration tests**: Test component interactions

## Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t agrodetect-ai .

# Run container
docker run -p 8501:8501 agrodetect-ai
```

### Cloud Deployment

The system supports deployment on:
- AWS (EC2, ECS, Lambda)
- Google Cloud Platform (Compute Engine, Cloud Run)
- Azure (Virtual Machines, Container Instances)

See deployment documentation for platform-specific instructions.

## Troubleshooting

### Common Issues

**Issue**: TensorFlow not installing
- **Solution**: Ensure you have Python 3.8-3.11. TensorFlow 2.15 doesn't support Python 3.12+

**Issue**: OpenCV import error
- **Solution**: Install system dependencies:
  - Ubuntu: `sudo apt-get install libgl1-mesa-glx`
  - Mac: `brew install opencv`

**Issue**: Streamlit not starting
- **Solution**: Check if port 8501 is available. Use `streamlit run src/ui/app.py --server.port 8502` for alternate port

**Issue**: Model loading fails
- **Solution**: Ensure model file exists in `models/` directory. Train a model first or download pre-trained weights.

## Performance Optimization

- Use GPU for training: Set `device='gpu'` in configuration
- Enable model caching in Streamlit: Use `@st.cache_resource`
- Optimize images: Resize large images before upload
- Batch predictions: Process multiple images together for better throughput

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use AgroDetect AI in your research, please cite:

```bibtex
@software{agrodetect2024,
  title={AgroDetect AI: Intelligent Plant Disease Classification},
  author={AgroDetect Team},
  year={2024},
  url={https://github.com/yourusername/agrodetect-ai}
}
```

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/agrodetect-ai/issues)
- Email: support@agrodetect.ai

## Acknowledgments

- MobileNetV2 architecture from TensorFlow/Keras
- Plant disease datasets from PlantVillage and other sources
- Streamlit for the amazing UI framework

---

**Version:** 0.1.0  
**Last Updated:** January 2024  
**Status:** Active Development
