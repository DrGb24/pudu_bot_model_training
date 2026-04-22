#!/usr/bin/env python3
"""
Updated LSTM Inference Engine
Uses new model trained on 147K partitioned data
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import json

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
import joblib
from tensorflow import keras
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LSTMInferencePartitioned:
    """
    LSTM Inference Engine using Partitioned Data Model
    
    Loads pre-trained model from 147K samples across train/val/test partitions.
    Provides real-time predictions with probability and risk categorization.
    """
    
    def __init__(self, 
                 model_path='models/lstm/lstm_partitioned.h5',
                 scaler_path='models/lstm/lstm_scaler_partitioned.pkl',
                 config_path='models/lstm/lstm_config_partitioned.json',
                 sequence_length=10):
        """
        Initialize inference engine with partitioned model
        
        Args:
            model_path: Trained model file path
            scaler_path: Feature normalization scaler
            config_path: Model configuration JSON
            sequence_length: Input sequence length (timesteps)
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path)
        self.sequence_length = sequence_length
        
        logger.info("Initializing LSTM Inference Engine (Partitioned Model)")
        logger.info(f"Model path: {self.model_path}")
        
        # Load model configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Model config loaded. Training date: {self.config.get('training_date')}")
        else:
            logger.warning("Config file not found, using defaults")
            self.config = {'sequence_length': sequence_length}
        
        # Build and load model
        self.model = self._build_model()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model weights from {self.model_path}")
        self.model.load_weights(str(self.model_path))
        logger.info("Model weights loaded")
        
        # Load scaler
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {self.scaler_path}")
        
        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        logger.info("Scaler loaded")
        
        logger.info("LSTM Inference Engine initialized successfully")
    
    def _build_model(self):
        """Build LSTM model architecture"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, 9)),
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            layers.LSTM(64, return_sequences=True, dropout=0.2),
            layers.LSTM(32, return_sequences=False, dropout=0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        return model
    
    def predict(self, sequences, return_proba=True):
        """
        Make predictions on sequences
        
        Args:
            sequences: Input data (samples, timesteps, features)
            return_proba: Return probability if True, binary if False
            
        Returns:
            Predictions
        """
        if not isinstance(sequences, np.ndarray):
            sequences = np.array(sequences).astype(np.float32)
        
        # Normalize
        n_samples = sequences.shape[0]
        timesteps = sequences.shape[1]
        features = sequences.shape[2]
        
        seq_reshaped = sequences.reshape(-1, features)
        seq_scaled = self.scaler.transform(seq_reshaped).reshape(n_samples, timesteps, features)
        
        predictions = self.model.predict(seq_scaled, verbose=0)
        
        if return_proba:
            return predictions
        else:
            return (predictions > 0.5).astype(int)
    
    def get_model_info(self):
        """Get model information"""
        return {
            'total_parameters': int(self.model.count_params()),
            'input_shape': (self.sequence_length, 9),
            'training_data_source': '147,488 samples (103K train, 22K val, 22K test)',
            'model_config': self.config
        }


def main():
    """Test inference engine"""
    print("\n" + "="*80)
    print("LSTM INFERENCE ENGINE - PARTITIONED MODEL")
    print("="*80)
    
    try:
        # Initialize engine
        engine = LSTMInferencePartitioned()
        
        # Show info
        info = engine.get_model_info()
        print("\nModel Information:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Training data: {info['training_data_source']}")
        print(f"  Config: {info['model_config']}")
        
        # Test with random sequence
        test_sequence = np.random.randn(1, 10, 9).astype(np.float32)
        prediction = engine.predict(test_sequence, return_proba=True)
        
        print(f"\nTest Prediction:")
        print(f"  Probability: {float(prediction[0][0]):.4f}")
        print(f"  Prediction: {'FAILURE' if prediction[0][0] > 0.5 else 'NORMAL'}")
        
        print("\n" + "="*80)
        print("Inference engine ready for production")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
