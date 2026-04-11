#!/usr/bin/env python3
"""
LSTM Inference Engine for Real-Time Predictions
Production-ready prediction system for robot failure detection
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lstm_models import LSTMInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'logs' / 'lstm' / 'inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / 'models' / 'lstm'

class LSTMPredictionEngine:
    """Real-time LSTM prediction engine"""
    
    def __init__(self):
        """Initialize inference engine"""
        
        model_path = MODEL_DIR / 'lstm_model.h5'
        scaler_path = MODEL_DIR / 'lstm_scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model not found: {model_path}")
        
        self.inference = LSTMInference(
            model_path=str(model_path),
            scaler_path=str(scaler_path) if scaler_path.exists() else None
        )
        
        # Load feature names
        features_path = MODEL_DIR / 'lstm_feature_names.npy'
        if features_path.exists():
            self.feature_names = np.load(features_path, allow_pickle=True)
        else:
            self.feature_names = None
        
        logger.info("✅ LSTM Prediction engine initialized")
    
    def predict_single_robot(self, 
                            robot_data: dict,
                            sequence_length: int = 10) -> dict:
        """
        Predict failure for single robot
        
        Args:
            robot_data: Dictionary with robot features
            sequence_length: Length of time-series sequence
            
        Returns:
            Prediction results
        """
        
        logger.info(f"📊 Predicting for robot: {robot_data.get('robot_id', 'Unknown')}")
        
        # Convert dict to array in correct feature order
        if self.feature_names:
            feature_values = np.array([
                robot_data.get(feat, 0) for feat in self.feature_names
            ])
        else:
            feature_values = np.array(list(robot_data.values()))
        
        # Create sequence (repeat last value to fill sequence)
        sequence = np.tile(feature_values, (sequence_length, 1))
        sequence = sequence.reshape(1, sequence_length, len(feature_values))
        
        # Make prediction
        result = self.inference.predict_single(sequence[0])
        
        logger.info(f"  Failure Probability: {result['probability']:.4f}")
        logger.info(f"  Prediction: {'⚠️  FAILURE' if result['prediction'] == 1 else '✅ NORMAL'}")
        logger.info(f"  Risk Level: {result['risk_level']}")
        
        return result
    
    def predict_batch(self,
                     sequences: np.ndarray) -> dict:
        """
        Batch predictions
        
        Args:
            sequences: Array of sequences (samples, timesteps, features)
            
        Returns:
            Batch prediction results
        """
        
        logger.info(f"📊 Batch predictions: {sequences.shape[0]} samples")
        
        result = self.inference.predict(sequences)
        
        return {
            'probabilities': result['probabilities'],
            'predictions': result['predictions'],
            'confidences': result['confidence'],
            'risk_levels': [
                self._categorize_risk(p) for p in result['probabilities']
            ]
        }
    
    @staticmethod
    def _categorize_risk(probability: float) -> str:
        """Categorize failure risk"""
        if probability < 0.3:
            return 'LOW'
        elif probability < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'


def example_single_prediction():
    """Example: Single robot prediction"""
    
    print("\n" + "="*70)
    print("ÖRNEK 1: TEK ROBOT TAHMINLEME")
    print("="*70 + "\n")
    
    engine = LSTMPredictionEngine()
    
    # Example robot data
    robot_data = {
        'error_count': 2,
        'task_hour': 14,
        'task_day_of_month': 15,
        'task_day_of_week': 3,
        'robot_id_length': 8,
        'software_version_length': 5,
        'product_code_type': 1,
        'error_severity': 2,
        'hourly_error_rate': 0.5
    }
    
    result = engine.predict_single_robot(robot_data)
    
    print(f"\n📋 Sonuç:")
    print(f"  Başarısızlık İhtimali: {result['probability']:.2%}")
    print(f"  Tahmin: {result['prediction']} (0=Normal, 1=Arıza)")
    print(f"  Güven: {result['confidence']:.4f}")
    print(f"  Risk Seviyesi: {result['risk_level']}")


def example_batch_prediction():
    """Example: Batch predictions"""
    
    print("\n" + "="*70)
    print("ÖRNEK 2: TOPLU TAHMINLEME")
    print("="*70 + "\n")
    
    engine = LSTMPredictionEngine()
    
    # Create random sequences (sample, timesteps=10, features=9)
    sequences = np.random.randn(5, 10, 9)
    
    results = engine.predict_batch(sequences)
    
    print(f"\n📊 Toplu Tahmin Sonuçları ({len(results['predictions'])} örnek):\n")
    print("  Örn. | Olasılık | Tahmin | Risk")
    print("  " + "─"*35)
    
    for i, (prob, pred, risk) in enumerate(zip(
        results['probabilities'],
        results['predictions'],
        results['risk_levels']
    )):
        pred_label = "⚠️  Arıza" if pred == 1 else "✅ Normal"
        print(f"  {i+1:<3} | {prob:>6.2%}   | {pred_label:14} | {risk}")


def main():
    """Main entry point"""
    
    try:
        example_single_prediction()
        example_batch_prediction()
        
    except FileNotFoundError as e:
        logger.error(f"Hata: {e}")
        logger.info("\nLSTM modeli henüz eğitilmedi. Önce lstm_train.py çalıştırın:")
        logger.info("  python lstm_train.py")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
