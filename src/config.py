"""
Configuration module for the project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_state': 42,
    'train_size': 0.7,       # 70% training
    'validation_size': 0.15, # 15% validation
    'test_size': 0.15,       # 15% test
}

# Data configuration
DATA_CONFIG = {
    'target_column': 'failure',
    'categorical_columns': [],
    'numerical_columns': [
        'error_count',
        'task_hour',
        'task_day_of_month',
        'task_day_of_week',
        'robot_id_length',
        'software_version_length',
        'product_code_type',
        'hourly_error_rate'
        # error_severity removed: derived from same error_level as failure target (data leakage)
    ],
}

# KPI target thresholds
KPI_TARGETS = {
    'prediction_accuracy': 0.95,  # Target: 95% (increased from 0.85)
    'recall': 0.85,
    'precision': 0.80,
    'f1_score': 0.80,
    'false_alarm_rate': 0.10,
    'system_latency': 60,  # seconds
    'system_uptime': 0.99,
    'error_handling_rate': 0.95,
    'connectivity_health': 0.95,
}

# LSTM Model Configuration
LSTM_CONFIG = {
    'sequence_length': 10,          # Timesteps in sequence
    'lstm_units': 64,               # Reduced from 128 (prevent overfitting)
    'dropout_rate': 0.4,            # Increased from 0.2
    'dense_units': 32,              # Reduced from 64
    'learning_rate': 0.0001,        # Keep low for stability
    'batch_size': 16,               # Reduced from 32 (more updates)
    'epochs': 100,                  # Max epochs
    'early_stopping_patience': 15,  # Early stopping patience
    'validation_split': 0.15,       # Validation portion
}

# Random Forest Configuration (for reference)
RF_CONFIG = {
    'n_estimators': 2000,
    'max_depth': 50,
    'criterion': 'entropy',
    'random_state': 42,
    'n_jobs': -1,
}

# HuggingFace dataset configuration
HUGGINGFACE_CONFIG = {
    'repo_id': 'Lightcap/pudu-robot-operation-logs-bau-capstone-2026',
    'config_name': 'partitioned_error_logs',
    # Splits: 'train' (103,241), 'validation' (22,123), 'test' (22,124)
}

# LSTM V2 — Multi-Output Model Configuration
LSTM_V2_CONFIG = {
    'sequence_length':         10,    # Timesteps per input sequence
    'lstm_units':             128,    # First LSTM layer units (second = units//2)
    'dropout_rate':           0.3,    # Dropout after each LSTM layer
    'dense_units':             64,    # Shared dense layer units
    'learning_rate':       0.0001,    # Adam learning rate
    'batch_size':              16,    # Training batch size
    'epochs':                 100,    # Maximum epochs (early stopping applies)
    'early_stopping_patience': 15,    # Patience for EarlyStopping
    'future_window':          168,    # Look-ahead hours for future failure targets (7 days)
}

# Financial parameters
FINANCIAL_CONFIG = {
    'cost_per_failure': 50000,  # USD
    'cost_per_hour_downtime': 5000,  # USD
    'system_cost': 500000,  # USD
    'maintenance_cost_per_robot': 10000,  # USD
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'formatter': 'standard',
            'filename': LOGS_DIR / 'predictive_maintenance.log'
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
