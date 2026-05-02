# Predictive Maintenance System - PUDU Robot LSTM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-brightgreen.svg)](https://scikit-learn.org/)
[![HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

Production-ready deep learning system for predictive maintenance of industrial robots, trained on **147,488 real robot error logs** sourced from [HuggingFace Datasets](https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026).

**LSTM Model**: 98.94% Accuracy | 95.76% Recall | 85.30% Precision | 99.33% AUC-ROC

---

## Quick Start

```bash
git clone https://github.com/DrGb24/pudu_bot_model_training.git
cd pudu_bot_model_training
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Train the Model
```bash
python lstm_train.py
```

### Run Inference
```bash
python lstm_inference_partitioned.py
```

---

## Model Performance

Evaluated on **22,124 held-out test samples** (from `robot_logs_error_test` table):

| Metric | Result | Target |
|--------|--------|--------|
| **Accuracy** | 98.94% | >=95% |
| **Recall** | 95.76% | >=85% |
| **Precision** | 85.30% | >=80% |
| **F1-Score** | 90.23% | >=80% |
| **AUC-ROC** | 99.33% | - |
| **Loss** | 0.0603 | - |

### Confusion Matrix (Test Set)
```
True Positives:   1,085   (failures correctly detected)
True Negatives:  20,795   (normal operation correctly identified)
False Positives:    187   (false alarms)
False Negatives:     48   (missed failures)
```

---

## Architecture

### LSTM Model
```
Input: 10 timesteps x 8 features
            |
Bidirectional LSTM Layer 1 (64 units, dropout=0.4)
            |
Bidirectional LSTM Layer 2 (32 units, dropout=0.4)
            |
Bidirectional LSTM Layer 3 (16 units, dropout=0.4)
            |
Dense Layer: 32 units, ReLU activation
            |
Output Layer: 1 unit, Sigmoid (failure probability)

Total Parameters: 33,505
Loss Function:    Binary Crossentropy
Optimizer:        Adam (learning_rate=0.0001)
Regularization:   Dropout (0.4)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Sequence Length | 10 timesteps |
| Batch Size | 16 |
| Max Epochs | 100 (early stopping patience=15) |
| Learning Rate | 0.0001 |
| Training Samples | 103,241 |
| Validation Samples | 22,123 |
| Test Samples | 22,124 |

---

## Dataset

All data is loaded from [HuggingFace Datasets](https://huggingface.co/datasets/Lightcap/pudu-robot-operation-logs-bau-capstone-2026). No synthetic data is used.

```
Dataset : Lightcap/pudu-robot-operation-logs-bau-capstone-2026
Config  : partitioned_error_logs
```

| Split | Rows | Purpose |
|-------|------|---------|
| `train` | 103,241 | Model training |
| `validation` | 22,123 | Hyperparameter tuning / early stopping |
| `test` | 22,124 | Final evaluation |
| **Total** | **147,488** | - |

### Input Features (8 features)
```
1. error_count              - Hourly error count
2. task_hour_num            - Hour of the task (0-23)
3. day_of_month             - Day of month (1-31)
4. day_of_week              - Day of week (0=Monday)
5. robot_id_length          - Length of robot identifier string
6. software_version_length  - Length of software version string
7. product_code_type        - Encoded product model (1-5)
8. hourly_error_rate        - Error rate per hour
```

### Target Variable
```
failure = 1  ->  error_level is "Error" or "Critical"
failure = 0  ->  error_level is "Event" or "Warning"
```

---

## Project Structure

```
project/
├── lstm_train.py                        Training pipeline (partitioned DB tables)
├── lstm_inference_partitioned.py        Inference engine
├── rf_train.py                          Random Forest training (backup model)
├── rf_inference.py                      Random Forest inference
│
├── src/
│   ├── config.py                        Centralized configuration & HuggingFace settings
│   ├── data_preparation.py              Data loading from HuggingFace Datasets
│   └── lstm_models.py                   LSTMModel class (build, train, evaluate, save)
│
├── models/
│   └── lstm/
│       ├── lstm_partitioned.h5          Trained model weights
│       ├── lstm_scaler_partitioned.pkl  StandardScaler (fitted on training data)
│       └── lstm_config_partitioned.json Model metadata & architecture config
│
├── logs/
│   └── lstm_training_report.txt         Training report with full metrics
│
└── requirements.txt
```

---

## Usage

### Training
```bash
python lstm_train.py
```

The training pipeline executes 7 steps:
1. Load data from HuggingFace (train / validation / test splits)
2. Feature engineering (23 raw columns to 8 features + binary target)
3. Create sequences (10 timesteps per sample)
4. Normalize features with StandardScaler
5. Build and train LSTM model
6. Evaluate on held-out test set
7. Save model, scaler, config, and report

### Inference
```python
from lstm_inference_partitioned import LSTMPartitionedInference

inference = LSTMPartitionedInference()

robot_data = {
    'error_count': 5,
    'task_hour_num': 14,
    'day_of_month': 22,
    'day_of_week': 1,
    'robot_id_length': 8,
    'software_version_length': 12,
    'product_code_type': 2,
    'hourly_error_rate': 0.08
}

result = inference.predict(robot_data)
print(f"Failure probability: {result['probability']:.4f}")
print(f"Risk level: {result['risk_level']}")
```

### Risk Levels
```
LOW:    Probability < 0.40   - Safe to operate
MEDIUM: 0.40 <= P < 0.70    - Monitor closely
HIGH:   Probability >= 0.70  - Maintenance urgent
```

---

## Dependencies

```
tensorflow>=2.13
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
joblib>=1.3
datasets>=2.14
huggingface_hub>=0.20
python-dotenv>=1.0
matplotlib>=3.7
seaborn>=0.12
```

Install: `pip install -r requirements.txt`

---

## Dataset Configuration

Dataset source is set in `src/config.py`:

```python
HUGGINGFACE_CONFIG = {
    'repo_id': 'Lightcap/pudu-robot-operation-logs-bau-capstone-2026',
    'config_name': 'partitioned_error_logs',
}
```

No database connection is required. Data is streamed directly from HuggingFace. Inference uses the saved model files only.
