<!-- Use this file to provide workspace-specific custom instructions to Copilot. -->

# Predictive Maintenance System - Custom Instructions

## Project Overview
This is a machine learning project for predictive maintenance of industrial robots using **Random Forest** binary classification model.

## Key Components
1. **Data Preparation** (`src/data_preparation.py`) - Data loading, cleaning, feature engineering, and preprocessing
2. **Random Forest Model** (`src/tree_models.py`) - Single RandomForestClassifier implementation (100 estimators, max_depth=15)
3. **KPI Metrics** (`src/kpi_metrics.py`) - 20+ performance metrics across 4 categories (model, operational, system, financial)
4. **Configuration** (`src/config.py`) - Centralized settings, KPI targets, and thresholds
5. **Training Pipeline** (`train.py`) - 6-step orchestration: data prep → model training → evaluation → KPI calculation → model storage → reporting
6. **Inference Engine** (`inference.py`) - Real-time prediction with risk scoring for new robot data

## Main Scripts
- `train.py` - 6-step training pipeline with comprehensive logging (Turkish named steps: ADIM 1-6)
- `inference.py` - Real-time prediction engine with probability and risk categorization
- `examples.py` - 5 practical examples covering full workflow
- `test_predictive_maintenance.py` - Unit tests for all modules

## Development Guidelines
- All Python code follows PEP 8 style guide
- Models use scikit-learn RandomForestClassifier interface
- Use RandomForestModel class from src/tree_models.py
- KPI thresholds defined in src/config.py (accuracy≥0.85, recall≥0.85, precision≥0.80)
- Log all operations to both file (logs/) and console
- Use absolute paths from src/config.py for file operations

## Model Architecture
- **Type**: Random Forest Binary Classifier
- **Target**: Robot failure prediction (0=normal, 1=failure)
- **Features**: 9 numerical sensor measurements
- **Training Data**: 2000 synthetic/real examples
- **Test Split**: 80/20 (train/test)
- **Hyperparameters**: n_estimators=100, max_depth=15, random_state=42

## Dependencies
- pandas, numpy, scikit-learn, joblib (core ML)
- matplotlib, seaborn (visualization)
- python-dotenv (configuration)

## Key Files Structure
```
project/
├── src/
│   ├── data_preparation.py     (DataPreparation class)
│   ├── tree_models.py          (RandomForestModel class)
│   ├── kpi_metrics.py          (KPIMetrics class)
│   └── config.py               (configuration constants)
├── train.py                    (Training orchestrator)
├── inference.py                (Prediction engine)
├── models/                     (Saved models directory)
├── logs/                       (Training logs & reports)
└── requirements.txt            (Python dependencies)
```

## Workflow
1. **Data Preparation**: Load, clean, scale, split data
2. **Model Training**: Fit Random Forest on 80% training data
3. **Evaluation**: Test on 20% test data, calculate metrics
4. **KPI Calculation**: Generate 20+ KPIs across 4 categories
5. **Model Persistence**: Save to models/random_forest_model.pkl
6. **Report Generation**: Create CSV reports with results

## Testing
- Synthetic data generation for quick validation
- Unit tests in test_predictive_maintenance.py
- Comprehensive KPI reports with targets
- Feature importance ranking
- Examples in examples.py

## Deployment
- Models saved to `models/` directory (joblib format)
- Inference available through RandomForestInference class
- Predictions include probability scores
- Risk categorization (LOW/MEDIUM/HIGH)
- Logging configured for production use

## Turkish Documentation
- **KURULUM_BASLAMA.md** - Quick start guide (3 setup methods, troubleshooting)
- **KURULUM_REHBERI.md** - Detailed installation steps
- **GENEL_ACIKLAMA.md** - System overview with flowcharts
- **DOSYALAR_OZET.md** - Per-file documentation
- **PYTHON_KURULUM_GEREKLI.md** - Python installation fixes

## Quick Start
```bash
# 1. Install Python 3.10+ (if needed, run: python_indir_kur.bat)
# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
./venv/Scripts/activate  # Windows
source venv/bin/activate # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt

# 5. Train the model
python train.py

# 6. Make predictions
python inference.py
```
