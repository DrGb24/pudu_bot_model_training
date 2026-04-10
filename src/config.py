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
        'error_severity',
        'hourly_error_rate'
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

# Database configuration (PostgreSQL / SQLite)
DATABASE_CONFIG = {
    'type': 'postgresql',  # 'postgresql' or 'sqlite'
    'host': '149.102.155.77',
    'port': 5433,
    'database': 'robot_pipeline',
    'user': 'robot_pipeline_admin',
    'password': 'RobotPipe!2026#PG!149',
    'ssl_mode': 'disable',
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
