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
    """Enhanced LSTM modeli ile inference yapan sınıf"""
    
    def __init__(self, weights_path='models/lstm/lstm_enhanced_focal.weights.h5',
                 scaler_path='models/lstm/lstm_scaler_enhanced.pkl',
                 sequence_length=10):
        """
        Args:
            weights_path: Eğitilmiş LSTM model weights
            scaler_path: Feature scaler
            sequence_length: Sequence uzunluğu
        """
        self.weights_path = Path(weights_path)
        self.scaler_path = Path(scaler_path)
        self.sequence_length = sequence_length
        
        # Model architecture'ı recreate et
        logger.info(f"🔨 Model architecture'ı yeniden oluşturuluyor...")
        self.model = self._build_model()
        
        # Weights yükle
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights bulunamadı: {self.weights_path}")
        
        logger.info(f"📦 Model weights yükleniyor: {self.weights_path}")
        self.model.load_weights(self.weights_path)
        logger.info(f"✅ Model weights yüklendi")
        
        # Scaler yükle
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler bulunamadı: {self.scaler_path}")
        
        logger.info(f"📦 Scaler yükleniyor: {self.scaler_path}")
        self.scaler = joblib.load(self.scaler_path)
        
        logger.info(f"✅ Model ve Scaler yüklendi")
        logger.info(f"   Model params: {self.model.count_params():,}")
        logger.info(f"   Sequence length: {self.sequence_length}")
    
    def _build_model(self):
        """Build LSTM architecture matching trained model"""
        model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, 9)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2)),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile (weights yüklendikten sonra çıkış için)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',  # Loss metadata için
            metrics=['accuracy']
        )
        
        return model
    
    def predict_from_dataframe(self, df, feature_columns=None):
        """
        DataFrame'den predictions yapma
        
        Args:
            df: Input dataframe (en az sequence_length satır)
            feature_columns: Kullanılacak feature sütunları
        
        Returns:
            dict with predictions
        """
        if feature_columns is None:
            feature_columns = ['error_count', 'task_hour', 'day_of_month', 'day_of_week',
                              'robot_id_length', 'software_version_length', 'product_code_type',
                              'error_severity', 'hourly_error_rate']
        
        if len(df) < self.sequence_length:
            raise ValueError(f"DataFrame'de en az {self.sequence_length} satır gerekli")
        
        # Features çıkart
        features = df[feature_columns].values
        
        # Normalize
        features_scaled = self.scaler.transform(features)
        
        # Son sequence'i al
        X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1).astype(np.float32)
        
        # Predict
        prob = self.model.predict(X, verbose=0)[0][0]
        
        return {
            'probability': float(prob),
            'prediction': int(prob >= 0.5),
            'risk_level': self._categorize_risk(prob),
            'last_sequence': df.tail(self.sequence_length)[feature_columns].to_dict('records')
        }
    
    def predict_batch(self, df_list, feature_columns=None):
        """
        Birden fazla robot için batch prediction
        
        Args:
            df_list: Robot başına DataFrame listesi
            feature_columns: Feature sütunları
        
        Returns:
            predictions dict
        """
        results = {}
        for robot_id, df in df_list.items():
            try:
                results[robot_id] = self.predict_from_dataframe(df, feature_columns)
            except Exception as e:
                logger.warning(f"⚠️ {robot_id} için prediction hatası: {e}")
                results[robot_id] = {'error': str(e)}
        
        return results
    
    @staticmethod
    def _categorize_risk(probability, threshold_low=0.3, threshold_mid=0.7):
        """
        Probability'i risk seviyesine dönüştür
        
        Args:
            probability: Model prediction probability (0-1)
            threshold_low: Low risk threshold
            threshold_mid: Medium risk threshold
        
        Returns:
            Risk seviyesi (LOW, MEDIUM, HIGH)
        """
        if probability < threshold_low:
            return "LOW"
        elif probability < threshold_mid:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def get_model_info(self):
        """Model bilgileri döndür"""
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
    """Demo predictions"""
    logger.info("\n" + "="*80)
    logger.info("DEMO: Example Predictions")
    logger.info("="*80)
    
    try:
        # Inference engine oluştur
        engine = LSTMEnhancedInference()
        
        # Model bilgileri
        info = engine.get_model_info()
        logger.info("\n📊 Model Bilgileri:")
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        # Örnek veri oluştur (DB'den veya simulated)
        logger.info(f"\n📊 Örnek tahminler yapılıyor...")
        
        # Normal robot örneği (simulation)
        df_normal = pd.DataFrame([
            {'error_count': i % 5, 'task_hour': (i * 2) % 24, 'day_of_month': i % 28 + 1,
             'day_of_week': i % 7, 'robot_id_length': 15, 'software_version_length': 5,
             'product_code_type': 1, 'error_severity': i % 3, 'hourly_error_rate': 0}
            for i in range(15)
        ])
        result_normal = engine.predict_from_dataframe(df_normal)
        logger.info(f"\n✅ Normal robot (tahmin):")
        logger.info(f"   Probability: {result_normal['probability']:.4f}")
        logger.info(f"   Prediction: {'FAILURE' if result_normal['prediction'] else 'NORMAL'}")
        logger.info(f"   Risk: {result_normal['risk_level']}")
        
        # Failure örneği (simulation)
        df_failure = pd.DataFrame([
            {'error_count': 15 + (i % 5), 'task_hour': (i * 3) % 24, 'day_of_month': i % 28 + 1,
             'day_of_week': i % 7, 'robot_id_length': 15, 'software_version_length': 5,
             'product_code_type': 2, 'error_severity': 8 + (i % 2), 'hourly_error_rate': 1}
            for i in range(15)
        ])
        result_failure = engine.predict_from_dataframe(df_failure)
        logger.info(f"\n⚠️ Failing robot (tahmin):")
        logger.info(f"   Probability: {result_failure['probability']:.4f}")
        logger.info(f"   Prediction: {'FAILURE' if result_failure['prediction'] else 'NORMAL'}")
        logger.info(f"   Risk: {result_failure['risk_level']}")
            
        logger.info("\n" + "="*80)
        logger.info("✅ DEMO TAMAMLANDI")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Demo hatası: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    demo_predictions()
