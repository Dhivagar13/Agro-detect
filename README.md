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
│   ├── models/            # Model architecture and training
│   ├── inference/         # Inference engine
│   ├── auth/              # Authentication services
│   └── ui/                # Streamlit interface
├── tests/                 # Test suite
├── data/                  # Data storage
│   ├── raw/              # Raw images
│   ├── processed/        # Processed datasets
│   └── manifests/        # Dataset manifests
├── models/                # Trained models
├── config/                # Configuration files
├── logs/                  # Application logs
└── requirements.txt       # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config/config.yaml config/config.local.yaml
# Edit config.local.yaml with your settings
```

## Usage

### Training a Model

```python
from src.models.disease_classifier import DiseaseClassifier
from src.models.training_manager import TrainingManager

# Initialize classifier
classifier = DiseaseClassifier(num_classes=10)
classifier.build_model()

# Train model
trainer = TrainingManager(classifier)
trainer.train(train_dataset, val_dataset)
```

### Running the Streamlit App

```bash
streamlit run src/ui/app.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test types
pytest -m unit
pytest -m property
pytest -m integration

# Run with coverage
pytest --cov=src --cov-report=html
```

## Development

### Code Style

This project follows PEP 8 style guidelines. Format code using:

```bash
black src/ tests/
flake8 src/ tests/
```

### Testing

- Unit tests: Test individual components
- Property-based tests: Test invariants using Hypothesis
- Integration tests: Test component interactions

## License

MIT License

## Contributors

AgroDetect Team
