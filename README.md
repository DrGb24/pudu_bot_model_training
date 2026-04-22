# Predictive Maintenance System - PUDU Robot LSTM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-brightgreen.svg)](https://scikit-learn.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Required-336791.svg)](https://www.postgresql.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)](#)

Production-ready deep learning system for predictive maintenance of industrial robots, trained on **147,488 real robot error logs** from a partitioned PostgreSQL database.

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

All data comes from a live PostgreSQL database. No synthetic data is used.

| Table | Rows | Purpose |
|-------|------|---------|
| `robot_logs_error_training` | 103,241 | Model training |
| `robot_logs_error_validation` | 22,123 | Hyperparameter tuning / early stopping |
| `robot_logs_error_test` | 22,124 | Final evaluation |
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
│   ├── config.py                        Centralized configuration & DB credentials
│   ├── data_preparation.py              Data loading from PostgreSQL
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
1. Load data from 3 partitioned database tables
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
psycopg2-binary>=2.9
python-dotenv>=1.0
matplotlib>=3.7
seaborn>=0.12
```

Install: `pip install -r requirements.txt`

---

## Database Configuration

Set credentials in `src/config.py`:

```python
DATABASE_CONFIG = {
    'type': 'postgresql',
    'host': '<host>',
    'port': 5433,
    'database': 'robot_pipeline',
    'user': '<user>',
    'password': '<password>',
}
```

A live database connection is **required** for training. Inference uses the saved model files and does not need a database connection.
