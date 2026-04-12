#!/usr/bin/env python3
"""
Inference with Enhanced LSTM Model
Real-time failure prediction using LSTM Enhanced (96.96% recall)
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
from lstm_models import LSTMInference
import joblib
from tensorflow import keras
from tensorflow.keras import layers

# Suppress Keras warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("LSTM ENHANCED INFERENCE ENGINE")
print("="*80)

class LSTMEnhancedInference:
    """
    LSTM-based inference engine for real-time failure prediction.
    Loads pre-trained bidirectional LSTM model with 348,993 parameters.
    Achieves 96.96% recall on validation dataset.
    """
    
    def __init__(self, weights_path='models/lstm/lstm_enhanced_focal.weights.h5',
                 scaler_path='models/lstm/lstm_scaler_enhanced.pkl',
                 sequence_length=10):
        """
        Initialize the inference engine with pre-trained model weights and scaler.
        
        Args:
            weights_path (str): Path to trained LSTM model weights file
            scaler_path (str): Path to feature normalization scaler
            sequence_length (int): Number of timesteps in input sequences (default: 10)
            
        Raises:
            FileNotFoundError: If model weights or scaler file not found
        """
        self.weights_path = Path(weights_path)
        self.scaler_path = Path(scaler_path)
        self.sequence_length = sequence_length
        
        # Reconstruct model architecture from saved configuration
        logger.info("Reconstructing LSTM model architecture...")
        self.model = self._build_model()
        
        # Load pre-trained weights
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")
        
        logger.info(f"Loading model weights from {self.weights_path}")
        self.model.load_weights(self.weights_path)
        logger.info("Model weights loaded successfully")
        
        # Load feature scaler for input normalization
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Feature scaler not found: {self.scaler_path}")
        
        logger.info(f"Loading feature scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        logger.info("Model and scaler initialized")
        logger.info(f"   Total parameters: {self.model.count_params():,}")
        logger.info(f"   Sequence length: {self.sequence_length}")
    
    def _build_model(self):
        """
        Build LSTM model architecture matching the trained configuration.
        
        Architecture:
        - 3 bidirectional LSTM layers (128, 64, 32 units)
        - Dropout regularization (0.2-0.3)
        - L2 weight regularization (0.001)
        - Sigmoid output for binary classification
        
        Returns:
            keras.Sequential: Compiled model instance
        """
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, 9)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2)),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with binary crossentropy loss and Adam optimizer
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict_from_dataframe(self, df, feature_columns=None):
        """
        Generate failure prediction for input time series data.
        
        Processes input dataframe by normalizing features and generating
        prediction using the trained LSTM model.
        
        Args:
            df (pd.DataFrame): Input data with at least sequence_length rows
            feature_columns (list): Names of feature columns (default: robot sensor columns)
        
        Returns:
            dict: Contains prediction results:
                - 'probability': Float failure probability (0-1)
                - 'prediction': Binary prediction (0=normal, 1=failure)
                - 'risk_level': Categorical risk assessment (LOW/MEDIUM/HIGH)
                - 'last_sequence': DataFrame of input sequence used
                
        Raises:
            ValueError: If dataframe has insufficient rows
        """
        if feature_columns is None:
            feature_columns = ['error_count', 'task_hour', 'day_of_month', 'day_of_week',
                              'robot_id_length', 'software_version_length', 'product_code_type',
                              'error_severity', 'hourly_error_rate']
        
        if len(df) < self.sequence_length:
            raise ValueError(f"Dataframe requires at least {self.sequence_length} rows")
        
        # Extract features as numpy array
        features = df[feature_columns].values
        
        # Normalize using fitted scaler
        features_scaled = self.scaler.transform(features)
        
        # Extract last sequence_length timesteps and reshape for model
        X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1).astype(np.float32)
        
        # Generate prediction probability
        prob = self.model.predict(X, verbose=0)[0][0]
        
        return {
            'probability': float(prob),
            'prediction': int(prob >= 0.5),
            'risk_level': self._categorize_risk(prob),
            'last_sequence': df.tail(self.sequence_length)[feature_columns].to_dict('records')
        }
    
    def predict_batch(self, df_list, feature_columns=None):
        """
        Generate batch predictions for multiple robot time series.
        
        Processes multiple dataframes and aggregates predictions.
        Handles individual errors gracefully.
        
        Args:
            df_list (dict): Mapping of robot IDs to DataFrame objects
            feature_columns (list): Feature column names
        
        Returns:
            dict: Mapping of robot IDs to prediction dictionaries
        """
        results = {}
        for robot_id, df in df_list.items():
            try:
                results[robot_id] = self.predict_from_dataframe(df, feature_columns)
            except Exception as e:
                logger.warning(f"Prediction error for {robot_id}: {e}")
                results[robot_id] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def _categorize_risk(probability, threshold_low=0.3, threshold_mid=0.7):
        """
        Categorize failure probability into risk levels.
        
        Classification thresholds based on operational requirements:
        - LOW: Probability < 0.3 (safe operation, routine maintenance)
        - MEDIUM: 0.3 <= Probability < 0.7 (monitor closely, increase diagnostics)
        - HIGH: Probability >= 0.7 (immediate maintenance required)
        
        Args:
            probability (float): Model output probability (0-1)
            threshold_low (float): Low-to-medium risk boundary (default: 0.3)
            threshold_mid (float): Medium-to-high risk boundary (default: 0.7)
        
        Returns:
            str: Risk level (LOW, MEDIUM, HIGH)
        """
        if probability < threshold_low:
            return "LOW"
        elif probability < threshold_mid:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def get_model_info(self):
        """
        Return comprehensive model metadata.
        
        Returns:
            dict: Model configuration and performance metrics
        """
        return {
            'model_type': 'LSTM Enhanced',
            'architecture': 'BiLSTM 3-layer',
            'parameters': self.model.count_params(),
            'loss_function': 'Focal Loss',
            'training_samples': 18860,
            'test_recall': 0.9696,
            'test_precision': 0.9253,
            'test_auc': 0.9968,
            'sequence_length': self.sequence_length,
            'features': 9
        }


def demo_predictions():
    """
    Execute demonstration predictions using simulated robot telemetry.
    
    Generates example predictions for:
    1. Normal operating robot with low error counts
    2. Robot approaching failure threshold with high errors
    
    Used for validation that inference engine is operational.
    """
    logger.info("\n" + "="*80)
    logger.info("DEMONSTRATION: Example Predictions")
    logger.info("="*80)
    
    try:
        # Initialize inference engine with trained model
        engine = LSTMEnhancedInference()
        
        # Log model configuration
        info = engine.get_model_info()
        logger.info("\nModel Configuration:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        # Generate example predictions using simulated data
        logger.info("\nGenerating example predictions...")
        
        # Simulate normal operating robot with minimal errors
        df_normal = pd.DataFrame([
            {'error_count': i % 5, 'task_hour': (i * 2) % 24, 'day_of_month': i % 28 + 1,
             'day_of_week': i % 7, 'robot_id_length': 15, 'software_version_length': 5,
             'product_code_type': 1, 'error_severity': i % 3, 'hourly_error_rate': 0}
            for i in range(15)
        ])
        result_normal = engine.predict_from_dataframe(df_normal)
        logger.info("\nNormal Robot Prediction:")
        logger.info(f"   Probability: {result_normal['probability']:.4f}")
        logger.info(f"   Prediction: {'FAILURE' if result_normal['prediction'] else 'NORMAL'}")
        logger.info(f"   Risk Level: {result_normal['risk_level']}")
        
        # Simulate robot with high fault indicators approaching failure
        df_failure = pd.DataFrame([
            {'error_count': 15 + (i % 5), 'task_hour': (i * 3) % 24, 'day_of_month': i % 28 + 1,
             'day_of_week': i % 7, 'robot_id_length': 15, 'software_version_length': 5,
             'product_code_type': 2, 'error_severity': 8 + (i % 2), 'hourly_error_rate': 1}
            for i in range(15)
        ])
        result_failure = engine.predict_from_dataframe(df_failure)
        logger.info("\nFailing Robot Prediction:")
        logger.info(f"   Probability: {result_failure['probability']:.4f}")
        logger.info(f"   Prediction: {'FAILURE' if result_failure['prediction'] else 'NORMAL'}")
        logger.info(f"   Risk Level: {result_failure['risk_level']}")
            
        logger.info("\n" + "="*80)
        logger.info("DEMONSTRATION COMPLETED")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    demo_predictions()
