#!/usr/bin/env python3
"""
LSTM Training Pipeline with Partitioned Database Tables
Train LSTM model using data from robot_logs_error_training,
robot_logs_error_validation, and robot_logs_error_test tables.
Total: ~147,000 samples across 3 data splits.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG, DATA_CONFIG, LSTM_CONFIG, HUGGINGFACE_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from sklearn.preprocessing import StandardScaler

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'lstm_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path('models/lstm')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 10

# These columns will be extracted/engineered from database columns
FEATURE_COLUMNS = [
    'error_count',              # From hourly_error_count
    'task_hour_num',            # Extracted from task_hour
    'day_of_month',             # Extracted from task_time
    'day_of_week',              # Extracted from task_time
    'robot_id_length',          # Calculated from robot_id
    'software_version_length',  # Calculated from soft_version
    'product_code_type',        # Encoded from product_code
    'hourly_error_rate'         # From hourly_ratio
    # NOTE: error_severity removed - it's derived from error_level same as failure target (data leakage)
]


class LSTMTrainer:
    """LSTM training pipeline using partitioned database tables"""
    
    def __init__(self):
        self.data_prep = DataPreparation()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = StandardScaler()
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load data from 3 partitioned tables"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING FROM PARTITIONED TABLES")
        logger.info("="*80)
        
        try:
            # Load training data
            logger.info("\nLoading training data from HuggingFace (split=train)...")
            self.train_data = self.data_prep.load_from_huggingface(
                HUGGINGFACE_CONFIG,
                split='train'
            )
            logger.info(f"   Training data shape: {self.train_data.shape}")
            logger.info(f"   Columns: {list(self.train_data.columns)}")

            # Load validation data
            logger.info("\nLoading validation data from HuggingFace (split=validation)...")
            self.val_data = self.data_prep.load_from_huggingface(
                HUGGINGFACE_CONFIG,
                split='validation'
            )
            logger.info(f"   Validation data shape: {self.val_data.shape}")
            
            # Load test data
            logger.info("\nLoading test data from HuggingFace (split=test)...")
            self.test_data = self.data_prep.load_from_huggingface(
                HUGGINGFACE_CONFIG,
                split='test'
            )
            logger.info(f"   Test data shape: {self.test_data.shape}")
            
            total_samples = len(self.train_data) + len(self.val_data) + len(self.test_data)
            logger.info(f"\nTotal samples across all splits: {total_samples:,}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_sequences(self, df, target_col='failure'):
        """Convert dataframe to sequences for LSTM"""
        logger.info(f"\nCreating sequences from {len(df)} samples...")
        
        features = []
        labels = []
        
        # Ensure required columns exist
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, creating with zeros")
                df[col] = 0
        
        # Create sequences
        for i in range(len(df) - SEQUENCE_LENGTH + 1):
            seq = df[FEATURE_COLUMNS].iloc[i:i+SEQUENCE_LENGTH].values
            features.append(seq)
            labels.append(df[target_col].iloc[i+SEQUENCE_LENGTH-1])
        
        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        logger.info(f"   Sequences created: {X.shape[0]}")
        logger.info(f"   Shape per sequence: {X.shape[1:]} (timesteps × features)")
        
        return X, y
    
    def engineer_features(self, df):
        """Create required features from raw database columns"""
        logger.info("Performing feature engineering...")
        
        df = df.copy()
        
        # Extract features from task_time
        if 'task_time' in df.columns:
            df['task_time'] = pd.to_datetime(df['task_time'])
            df['day_of_month'] = df['task_time'].dt.day
            df['day_of_week'] = df['task_time'].dt.dayofweek
        
        # Extract hour from task_hour
        if 'task_hour' in df.columns:
            df['task_hour'] = pd.to_datetime(df['task_hour'])
            df['task_hour_num'] = df['task_hour'].dt.hour
        
        # Rename/map columns
        if 'hourly_error_count' in df.columns:
            df['error_count'] = df['hourly_error_count']
        
        if 'hourly_ratio' in df.columns:
            df['hourly_error_rate'] = df['hourly_ratio']
        
        # Calculate robot_id_length
        if 'robot_id' in df.columns:
            df['robot_id_length'] = df['robot_id'].astype(str).str.len()
        
        # Calculate software_version_length
        if 'soft_version' in df.columns:
            df['software_version_length'] = df['soft_version'].astype(str).str.len()
        
        # Encode product_code_type
        if 'product_code' in df.columns:
            product_mapping = {
                'PuduBot2': 1,
                'KettyBot': 2,
                'Bellabot': 3,
                'CC': 4
            }
            df['product_code_type'] = df['product_code'].map(product_mapping).fillna(5).astype(int)
        
        # Encode error_level as error_severity (severity: Event=0, Warning=1, Error=2, Critical=3)
        if 'error_level' in df.columns:
            severity_mapping = {
                'Event': 0,
                'Warning': 1,
                'Error': 2,
                'Critical': 3
            }
            df['error_severity'] = df['error_level'].map(severity_mapping).fillna(0).astype(int)
        
        # Create binary target: failure indicator (1 if error_level is Error/Critical, else based on error_detail)
        if 'error_level' in df.columns:
            # Critical and error-level records are failures
            df['failure'] = ((df['error_level'].isin(['Error', 'Critical']))).astype(int)
        else:
            df['failure'] = 1  # Default to failure if present
        
        logger.info(f"Feature engineering complete. Failure distribution: {df['failure'].value_counts().to_dict()}")
        
        return df
    
    def prepare_data(self):
        """Prepare sequences and features"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        try:
            # Feature engineering
            self.train_data = self.engineer_features(self.train_data)
            self.val_data = self.engineer_features(self.val_data)
            self.test_data = self.engineer_features(self.test_data)
            
            logger.info("\n" + "="*80)
            logger.info("STEP 3: SEQUENCE CREATION")
            logger.info("="*80)
            
            # Create sequences
            logger.info("\n[Training Data]")
            X_train, y_train = self.create_sequences(self.train_data)
            
            logger.info("\n[Validation Data]")
            X_val, y_val = self.create_sequences(self.val_data)
            
            logger.info("\n[Test Data]")
            X_test, y_test = self.create_sequences(self.test_data)
            
            # Normalize features
            logger.info("\nNormalizing features with StandardScaler...")
            
            # Reshape for scaling
            n_train, ts, features = X_train.shape
            X_train_flat = X_train.reshape(-1, features)
            
            # Check for NaN/Inf before scaling
            logger.info(f"NaN count before scaling: {np.isnan(X_train_flat).sum()}")
            logger.info(f"Inf count before scaling: {np.isinf(X_train_flat).sum()}")
            
            # Fit scaler on training data
            self.scaler.fit(X_train_flat)
            
            # Transform safely
            X_train_scaled = self.scaler.transform(X_train_flat).reshape(n_train, ts, features)
            logger.info(f"NaN count after scaling train: {np.isnan(X_train_scaled).sum()}")
            
            # Handle any NaN values
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Validation
            n_val = X_val.shape[0]
            X_val_flat = X_val.reshape(-1, features)
            X_val_scaled = self.scaler.transform(X_val_flat).reshape(n_val, ts, features)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Test
            n_test = X_test.shape[0]
            X_test_flat = X_test.reshape(-1, features)
            X_test_scaled = self.scaler.transform(X_test_flat).reshape(n_test, ts, features)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logger.info("   Scaling complete")
            
            # Data statistics
            logger.info("\n" + "-"*80)
            logger.info("DATA STATISTICS")
            logger.info("-"*80)
            
            logger.info(f"\nTraining Data:")
            logger.info(f"   Sequences: {X_train_scaled.shape[0]:,}")
            logger.info(f"   Failures (1): {np.sum(y_train)} ({100*np.sum(y_train)/len(y_train):.2f}%)")
            logger.info(f"   Normal (0): {len(y_train) - np.sum(y_train)} ({100*(len(y_train)-np.sum(y_train))/len(y_train):.2f}%)")
            
            logger.info(f"\nValidation Data:")
            logger.info(f"   Sequences: {X_val_scaled.shape[0]:,}")
            logger.info(f"   Failures (1): {np.sum(y_val)} ({100*np.sum(y_val)/len(y_val):.2f}%)")
            logger.info(f"   Normal (0): {len(y_val) - np.sum(y_val)} ({100*(len(y_val)-np.sum(y_val))/len(y_val):.2f}%)")
            
            logger.info(f"\nTest Data:")
            logger.info(f"   Sequences: {X_test_scaled.shape[0]:,}")
            logger.info(f"   Failures (1): {np.sum(y_test)} ({100*np.sum(y_test)/len(y_test):.2f}%)")
            logger.info(f"   Normal (0): {len(y_test) - np.sum(y_test)} ({100*(len(y_test)-np.sum(y_test))/len(y_test):.2f}%)")
            
            return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: BUILD LSTM MODEL")
        logger.info("="*80)
        
        lstm_config = LSTM_CONFIG
        
        self.model = LSTMModel(
            input_shape=input_shape,
            lstm_units=lstm_config['lstm_units'],
            dropout_rate=lstm_config['dropout_rate'],
            dense_units=lstm_config['dense_units'],
            learning_rate=lstm_config['learning_rate']
        )
        
        self.model.build_model()
        logger.info(f"\nModel built successfully")
        logger.info(f"   Total parameters: {self.model.model.count_params():,}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train LSTM model"""
        logger.info("\n" + "="*80)
        logger.info("STEP 5: TRAIN MODEL")
        logger.info("="*80)
        
        # Calculate class weights for imbalanced data
        n_class_0 = np.sum(y_train == 0)
        n_class_1 = np.sum(y_train == 1)
        
        class_weights = {
            0: 1.0,
            1: n_class_0 / max(n_class_1, 1)  # Weight failures more heavily
        }
        
        logger.info(f"\nClass weights calculated:")
        logger.info(f"   Class 0 (Normal): {class_weights[0]:.4f}")
        logger.info(f"   Class 1 (Failure): {class_weights[1]:.4f}")
        
        logger.info(f"\nTraining parameters:")
        logger.info(f"   Training samples: {len(y_train):,}")
        logger.info(f"   Validation samples: {len(y_val):,}")
        logger.info(f"   Epochs: {LSTM_CONFIG['epochs']}")
        logger.info(f"   Batch size: {LSTM_CONFIG['batch_size']}")
        
        self.history = self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=LSTM_CONFIG['epochs'],
            batch_size=LSTM_CONFIG['batch_size'],
            verbose=1,
            class_weight=class_weights
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model on test data"""
        logger.info("\n" + "="*80)
        logger.info("STEP 6: EVALUATE MODEL")
        logger.info("="*80)
        
        try:
            metrics = self.model.evaluate(X_test, y_test)
            
            logger.info(f"\nTest Set Performance:")
            logger.info(f"   Loss: {metrics.get('loss', 'N/A')}")
            logger.info(f"   Accuracy: {metrics.get('accuracy', 0.0):.4f}")
            logger.info(f"   Precision: {metrics.get('precision', 0.0):.4f}")
            logger.info(f"   Recall: {metrics.get('recall', 0.0):.4f}")
            logger.info(f"   AUC: {metrics.get('auc', 0.0):.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            # Return default metrics if error
            return {
                'loss': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': 0.0
            }
    
    def save_model(self):
        """Save model and scaler"""
        logger.info("\n" + "="*80)
        logger.info("STEP 7: SAVE MODEL & ARTIFACTS")
        logger.info("="*80)
        
        try:
            # Save model weights
            model_path = MODEL_DIR / 'lstm_partitioned.h5'
            self.model.save_model(str(model_path))
            logger.info(f"\nModel weights saved: {model_path}")
            
            # Save scaler
            scaler_path = MODEL_DIR / 'lstm_scaler_partitioned.pkl'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Feature scaler saved: {scaler_path}")
            
            # Save model config
            config_path = MODEL_DIR / 'lstm_config_partitioned.json'
            import json
            config = {
                'input_shape': self.model.input_shape,
                'lstm_units': self.model.lstm_units,
                'dropout_rate': self.model.dropout_rate,
                'dense_units': self.model.dense_units,
                'learning_rate': self.model.learning_rate,
                'sequence_length': SEQUENCE_LENGTH,
                'feature_columns': FEATURE_COLUMNS,
                'training_date': datetime.now().isoformat()
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Model config saved: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def generate_report(self, metrics):
        """Generate training report"""
        logger.info("\n" + "="*80)
        logger.info("TRAINING REPORT")
        logger.info("="*80)
        
        report = f"""
LSTM Model Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATA SOURCES
============
- Training Table: robot_logs_error_training
- Validation Table: robot_logs_error_validation
- Test Table: robot_logs_error_test

DATASET STATISTICS
==================
Training sequences: {len(self.train_data):,}
Validation sequences: {len(self.val_data):,}
Test sequences: {len(self.test_data):,}
Total samples: {len(self.train_data) + len(self.val_data) + len(self.test_data):,}

SEQUENCE PARAMETERS
===================
Sequence length: {SEQUENCE_LENGTH} timesteps
Feature count: {len(FEATURE_COLUMNS)}
Features: {', '.join(FEATURE_COLUMNS)}

MODEL ARCHITECTURE
==================
LSTM units: {LSTM_CONFIG['lstm_units']}
Dropout rate: {LSTM_CONFIG['dropout_rate']}
Dense units: {LSTM_CONFIG['dense_units']}
Learning rate: {LSTM_CONFIG['learning_rate']}

TEST PERFORMANCE
================
Loss: {metrics.get('loss', 'N/A')}
Accuracy: {metrics.get('accuracy', 0.0):.4f} ({100*metrics.get('accuracy', 0.0):.2f}%)
Precision: {metrics.get('precision', 0.0):.4f}
Recall: {metrics.get('recall', 0.0):.4f}
F1-Score: {metrics.get('f1_score', 0.0):.4f}
AUC: {metrics.get('auc', 0.0):.4f}
True Positives:  {metrics.get('true_positives', 'N/A')}
True Negatives:  {metrics.get('true_negatives', 'N/A')}
False Positives: {metrics.get('false_positives', 'N/A')}
False Negatives: {metrics.get('false_negatives', 'N/A')}

ARTIFACTS SAVED
===============
- Model weights: {MODEL_DIR / 'lstm_partitioned.h5'}
- Feature scaler: {MODEL_DIR / 'lstm_scaler_partitioned.pkl'}
- Model config: {MODEL_DIR / 'lstm_config_partitioned.json'}
"""
        
        logger.info(report)
        
        # Save report
        report_path = LOG_DIR / 'lstm_training_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"\nReport saved: {report_path}")
        
        return report


def main():
    """Main training pipeline"""
    
    logger.info("\n" + "█"*80)
    logger.info("█ LSTM TRAINING PIPELINE WITH PARTITIONED DATABASE TABLES")
    logger.info("█ Predictive Maintenance System - PUDU Robot")
    logger.info("█"*80)
    
    try:
        trainer = LSTMTrainer()
        
        # Step 1: Load data
        trainer.load_data()
        
        # Step 2: Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = trainer.prepare_data()
        
        # Step 3: Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        trainer.build_model(input_shape)
        
        # Step 4: Train model
        trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Step 5: Evaluate model
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Step 6: Save model
        trainer.save_model()
        
        # Generate report
        trainer.generate_report(metrics)
        
        logger.info("\n" + "█"*80)
        logger.info("█ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("█"*80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n{'█'*80}")
        logger.error("█ TRAINING FAILED")
        logger.error(f"█ Error: {str(e)}")
        logger.error(f"{'█'*80}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
