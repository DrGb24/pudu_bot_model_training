# LSTM Model Training Summary

## Project Architecture
Predictive maintenance system with two parallel models:
- **Random Forest (tree-based)**: Established, production-ready
- **LSTM (deep learning)**: New, time-series focused

## LSTM Model Specifications

### Architecture
- **Type**: Bidirectional LSTM with time-series features
- **Input**: Sequences of sensor measurements (timesteps × features)
- **Layers**:
  - LSTM Layer 1: 128 units + ReLU + 20% Dropout
  - LSTM Layer 2: 64 units + ReLU + 20% Dropout
  - Dense Layer 1: 64 units + ReLU + 10% Dropout
  - Dense Layer 2: 32 units + ReLU + 10% Dropout
  - Output: 1 unit + Sigmoid (binary classification)

### Hyperparameters
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Crossentropy
- **Sequence Length**: 10 timesteps
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

### Data Split
- **Training**: 70% (1400 samples)
- **Validation**: 15% (300 samples)
- **Test**: 15% (300 samples)

## Input Features (9 features)
1. error_count - Number of errors in task window
2. task_hour - Hour of day (0-23)
3. task_day_of_month - Day of month (1-31)
4. task_day_of_week - Day of week (0-6)
5. robot_id_length - Length of robot identifier
6. software_version_length - Software version string length
7. product_code_type - Robot product type category
8. error_severity - Error count severity indicator
9. hourly_error_rate - Errors per hour

## Data Source
- **Database**: PostgreSQL (robot_pipeline)
- **Table**: robot_logs_info (with robot_logs_error joins)
- **Records**: 2,000 samples
- **Target**: is_success flag (inverted to failure label)
- **Failure Rate**: 2.4% (48 failures, 1952 normal)

## Training Process
1. **Data Preparation**: Load from PostgreSQL, create sequences
2. **Model Building**: Construct LSTM architecture  
3. **Training**: Fit on sequential data with validation
4. **Evaluation**: Test on held-out test set
5. **KPI Calculation**: Compute performance metrics
6. **Model Saving**: Export to .h5 format + scaler
7. **Reporting**: Generate CSV reports

## Expected Performance
- **Accuracy Target**: ≥85%
- **Recall Target**: ≥85% (capture failures)
- **Precision Target**: ≥80%
- **AUC-ROC Target**: ≥0.90

## Project Structure
```
project/
├── src/
│   ├── lstm_models.py           (LSTMModel, LSTMInference classes)
│   ├── rf_models.py             (RandomForestModel class)
│   ├── data_preparation.py      (DataPreparation - shared)
│   ├── kpi_metrics.py           (KPIMetrics - shared)
│   └── config.py                (Configuration - shared)
│
├── models/
│   ├── lstm/
│   │   ├── lstm_model.h5        (Trained LSTM weights)
│   │   ├── lstm_scaler.pkl      (Feature scaler)
│   │   └── lstm_feature_names.npy
│   │
│   └── random_forest/
│       ├── random_forest_model.pkl
│       └── feature_names.npy
│
├── logs/
│   ├── lstm/
│   │   ├── training_*.log
│   │   ├── lstm_final_report.csv
│   │   └── training_history.json
│   │
│   └── random_forest/
│       ├── final_report.csv
│       ├── kpi_report.csv
│       └── training.log
│
├── lstm_train.py                (LSTM training pipeline)
├── lstm_inference.py            (LSTM prediction engine)
├── rf_train.py                  (Random Forest training)
├── rf_inference.py              (Random Forest prediction)
├── examples_lstm.py             (LSTM usage examples)
└── test_lstm.py                 (LSTM unit tests)
```

## Files to Create/Update
- [x] src/lstm_models.py - LSTM model classes
- [x] lstm_train.py - Training orchestration
- [x] lstm_inference.py - Inference engine  
- [ ] examples_lstm.py - Usage examples
- [ ] test_lstm.py - Unit tests
- [ ] README.md - Update with LSTM info

## Comparison: Random Forest vs LSTM

| Aspect | Random Forest | LSTM |
|--------|---------------|------|
| **Type** | Tree-based ensemble | Recurrent neural network |
| **Inputs** | Individual feature vectors | Time-series sequences |
| **Accuracy** | 99.33% | TBD (expected 90-95%) |
| **Recall** | 71.43% | TBD (target 85%+) |
| **Speed** | Fast inference (~5ms) | Slower (~50ms per sequence) |
| **Training Time** | ~2 seconds | ~5-10 minutes |
| **Interpretability** | Feature importance | Black box (less clear) |
| **Hardware** | CPU suitable | GPU recommended |
| **Use Case** | Categorical features | Sequential/temporal patterns |

## Next Steps
1. Create examples_lstm.py with sample predictions
2. Create test_lstm.py with unit tests
3. Update README.md with LSTM section
4. Train LSTM model and verify performance
5. Compare RF vs LSTM results
6. Commit to GitHub (awaiting approval)
