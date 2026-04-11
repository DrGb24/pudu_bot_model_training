"""
Random Forest Tahmin Motoru
Eğitilmiş modeli kullanarak yeni veriler için tahmin yapma
"""

import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestInference:
    """Random Forest tahmin motoru"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_model()
        self.load_scaler()
        
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Özellik adlarını yükle
        features_path = MODELS_DIR / 'feature_names.npy'
        if features_path.exists():
            self.feature_names = np.load(features_path, allow_pickle=True)
        
        logger.info(f"✅ Model yüklendi")
        logger.info(f"   Özellikler: {len(self.feature_names) if self.feature_names is not None else 'Bilinmiyor'}")
        
    def load_scaler(self):
        """Veri ölçeklemesi için scaler'ı yükle"""
        
        try:
            from src.data_preparation import DataPreparation
            self.scaler = DataPreparation().scaler
        except:
            self.scaler = None
    
    def predict(self, X):
        """
        Tahmin yap
        
        Parameters:
        - X: DataFrame veya numpy array
        
        Returns:
        - predictions (0 = Arıza yok, 1 = Arıza var)
        - probabilities (olasılık değerleri)
        """
        
        if isinstance(X, pd.DataFrame):
            # Doğru sütun sırası
            if self.feature_names is not None:
                X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_failure_risk(self, X):
        """
        Her mühendislik için arıza riskini tahmin et
        
        Returns:
        - DataFrame: tahmin ve risk skorları
        """
        
        predictions, probabilities = self.predict(X)
        
        # Sonuçlar dataframe'i
        results = pd.DataFrame({
            'prediction': ['Arıza Var' if p == 1 else 'Normal' for p in predictions],
            'no_failure_prob': probabilities[:, 0],
            'failure_prob': probabilities[:, 1],
            'risk_score': probabilities[:, 1],
            'risk_level': pd.cut(probabilities[:, 1], 
                                bins=[0, 0.3, 0.6, 1.0],
                                labels=['DÜŞÜK', 'ORTA', 'YÜKSEK'])
        })
        
        return results


def example_inference():
    """Tahmin örneği"""
    
    logger.info("="*60)
    logger.info("RANDOM FOREST TAHMİN ÖRNEĞİ")
    logger.info("="*60)
    
    # Tahmin motorunu başlat
    inference = RandomForestInference()
    
    # Örnek veriler (eğitim verisiyle aynı özellikler)
    sample_data = pd.DataFrame({
        'temperature': [75, 85, 90],
        'vibration': [0.4, 0.6, 0.9],
        'pressure': [95, 110, 125],
        'humidity': [40, 50, 60],
        'operational_hours': [1000, 5000, 8000],
        'error_count': [2, 5, 12],
        'last_maintenance_days': [100, 250, 350],
        'robot_age_months': [12, 60, 90],
        'power_consumption': [450, 550, 650],
    })
    
    logger.info("\n📊 GİRİŞ VERİLERİ:")
    logger.info(sample_data.to_string())
    
    # Tahmin yap
    try:
        results = inference.predict_failure_risk(sample_data)
        
        logger.info("\n🎯 TAHMİN SONUÇLARI:")
        logger.info(results.to_string())
        
        # Her robot için ayrıntılı bilgi
        logger.info("\n📈 AYRINTI:")
        for idx, row in results.iterrows():
            risk = row['risk_level']
            prob = row['failure_prob']
            logger.info(f"   Robot {idx}: {risk} Risk (%{prob*100:.1f} olasılık)")
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {str(e)}")


if __name__ == '__main__':
    try:
        example_inference()
    except FileNotFoundError as e:
        logger.error(f"❌ Hata: {str(e)}")
        logger.info("   Önce modeli eğitmek için: python train.py")
