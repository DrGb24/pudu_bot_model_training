#!/usr/bin/env python3
"""
LSTM (Long Short-Term Memory) Model for Predictive Maintenance
TensorFlow/Keras implementation for time-series failure prediction
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json

# Setup logging
logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM Neural Network for Robot Failure Prediction"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int],
                 lstm_units: int = 128,
                 dropout_rate: float = 0.2,
                 dense_units: int = 64,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model
        
        Args:
            input_shape: (sequence_length, num_features)
            lstm_units: Number of LSTM cells
            dropout_rate: Dropout rate for regularization
            dense_units: Units in dense layers
            learning_rate: Adam optimizer learning rate
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        
        self.model = None
        self.history = None
        
        logger.info(f"LSTM Model initialized: {input_shape} → {lstm_units} LSTM units")
    
    def build_model(self) -> Model:
        """Build LSTM architecture"""
        
        logger.info("Building LSTM architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First LSTM layer with dropout
            layers.LSTM(
                self.lstm_units, 
                activation='relu',
                return_sequences=True,
                name='lstm_1'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Second LSTM layer 
            layers.LSTM(
                self.lstm_units // 2,
                activation='relu',
                return_sequences=False,
                name='lstm_2'
            ),
            layers.Dropout(self.dropout_rate),
            
            # Dense layers
            layers.Dense(self.dense_units, activation='relu', name='dense_1'),
            layers.Dropout(self.dropout_rate / 2),
            
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dropout(self.dropout_rate / 2),
            
            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        
        logger.info(f"✅ LSTM Model built successfully")
        logger.info(f"   - Total parameters: {model.count_params():,}")
        
        return model
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: int = 1) -> dict:
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences (samples, timesteps, features)
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")
        
        logger.info(f"🚀 Starting LSTM training...")
        logger.info(f"   Training samples: {X_train.shape[0]}")
        logger.info(f"   Sequence length: {X_train.shape[1]}")
        logger.info(f"   Features: {X_train.shape[2]}")
        logger.info(f"   Epochs: {epochs} | Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info(f"✅ Training completed")
        
        return self.history.history
    
    def evaluate(self, 
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test sequences
            y_test: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        
        logger.info("📊 Evaluating LSTM model on test set...")
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        logger.info("\n" + "="*50)
        logger.info("LSTM MODEL EVALUATION")
        logger.info("="*50)
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  AUC-ROC:   {auc:.4f}")
        logger.info("="*50)
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input sequences (samples, timesteps, features)
            
        Returns:
            Predictions (probabilities)
        """
        
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath: str):
        """Save model weights to file"""
        
        if self.model is None:
            raise RuntimeError("Model not built.")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        logger.info(f"✅ Model saved: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights from file"""
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"✅ Model loaded: {filepath}")
    
    def get_feature_importance(self, feature_names: list = None) -> dict:
        """
        Get feature importance from LSTM weights
        This is approximate - LSTM doesn't have direct feature importance like trees
        """
        
        if self.model is None:
            raise RuntimeError("Model not built.")
        
        # Get input weights from first LSTM layer
        lstm_layer = self.model.get_layer('lstm_1')
        weights = lstm_layer.get_weights()
        input_weights = weights[0]  # (features, lstm_units*4)
        
        # Aggregate importance across LSTM gates
        importance = np.mean(np.abs(input_weights), axis=1)
        importance = importance / importance.sum()
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        importance_dict = {
            name: float(imp) 
            for name, imp in zip(feature_names, importance)
        }
        
        return importance_dict
    
    def get_config(self) -> dict:
        """Get model configuration"""
        
        return {
            'input_shape': self.input_shape,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'dense_units': self.dense_units,
            'learning_rate': self.learning_rate
        }


class LSTMInference:
    """Inference engine for LSTM predictions"""
    
    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to saved LSTM model
            scaler_path: Path to saved data scaler
        """
        
        self.model = keras.models.load_model(model_path)
        self.scaler = None
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"✅ LSTM Inference engine initialized")
    
    def predict(self, X: np.ndarray) -> dict:
        """
        Make prediction with confidence
        
        Args:
            X: Input sequences (samples, timesteps, features)
            
        Returns:
            Prediction results dictionary
        """
        
        predictions = self.model.predict(X, verbose=0)
        
        return {
            'probabilities': predictions.flatten(),
            'predictions': (predictions > 0.5).astype(int).flatten(),
            'confidence': np.max(np.abs(predictions - 0.5) * 2, axis=1).flatten()
        }
    
    def predict_single(self, X_single: np.ndarray) -> dict:
        """
        Predict for single sample
        
        Args:
            X_single: Single sequence (timesteps, features)
            
        Returns:
            Prediction result for single sample
        """
        
        X = X_single.reshape(1, X_single.shape[0], X_single.shape[1])
        result = self.predict(X)
        
        return {
            'probability': float(result['probabilities'][0]),
            'prediction': int(result['predictions'][0]),
            'confidence': float(result['confidence'][0]),
            'risk_level': self._categorize_risk(result['probabilities'][0])
        }
    
    @staticmethod
    def _categorize_risk(probability: float) -> str:
        """Categorize failure risk level"""
        
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
