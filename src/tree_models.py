"""
Random Forest Model for Predictive Maintenance
Industrial robot failure prediction system
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest model for predictive maintenance"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.trained = False
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize Random Forest classifier - Aggressive tuning for 95% accuracy target"""
        
        # Random Forest - Maximum performance tuning for 95%+ accuracy
        self.model = RandomForestClassifier(
            n_estimators=2000,         # 2000 trees - maximum ensemble strength
            max_depth=50,              # Very deep trees for complex patterns
            min_samples_split=2,       # Allow very fine-grained splits
            min_samples_leaf=1,        # Pure leaves - overfitting allowed for 95% target
            max_features='log2',       # Use log2 for even more diversity
            criterion='entropy',       # Entropy for more aggressive splits
            random_state=self.random_state,
            n_jobs=-1,                 # All CPU cores
            class_weight='balanced_subsample',
            oob_score=True,
            bootstrap=True,
            max_samples=1.0,
        )
        
        logger.info("Random Forest Model: 2000 ağaç, max_depth=50 (95% Accuracy Target)")
    
    def train(self, X_train, y_train):
        """
        Modeli eğit
        
        Parameters:
        - X_train: Eğitim verileri
        - y_train: Eğitim hedef değerleri
        """
        
        logger.info("Random Forest modeli eğitiliyor...")
        start_time = datetime.now()
        
        self.model.fit(X_train, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        self.trained = True
        
        logger.info(f"✅ Model eğitimi tamamlandı ({training_time:.2f} saniye)")
        logger.info(f"   - OOB (Out-of-Bag) Skoru: {self.model.oob_score_:.4f}")
        
    def predict(self, X):
        """
        Tahmin yap (0 = Arıza yok, 1 = Arıza var)
        
        Parameters:
        - X: Tahmin yapılacak veriler
        
        Returns:
        - predictions (0 veya 1)
        """
        
        if not self.trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Tahmin olasılığını ver
        
        Returns:
        - 2D array: [[0'ın olasılığı, 1'in olasılığı], ...]
        """
        
        if not self.trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        """
        En önemli özellikleri bul
        
        Parameters:
        - feature_names: Özellik adları
        
        Returns:
        - DataFrame: Özellikler ve önemi
        """
        
        if not self.trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def evaluate(self, X_test, y_test):
        """
        Modeli test setinde değerlendir
        
        Returns:
        - dict: accuracy, precision, recall, f1_score
        """
        
        if not self.trained:
            raise ValueError("Model henüz eğitilmemiş!")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        evaluation = {
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'classification_report': report,
        }
        
        logger.info(f"\n{'='*50}")
        logger.info(f"RANDOM FOREST DEĞERLENDİRMESİ")
        logger.info(f"{'='*50}")
        logger.info(f"  Doğruluk (Accuracy):  {evaluation['accuracy']:.4f} (Hedef: ≥0.85)")
        logger.info(f"  Kesinlik (Precision): {evaluation['precision']:.4f} (Hedef: ≥0.80)")
        logger.info(f"  Hatırlanma (Recall):  {evaluation['recall']:.4f} (Hedef: ≥0.85)")
        logger.info(f"  F1-Score:             {evaluation['f1_score']:.4f} (Hedef: ≥0.80)")
        logger.info(f"{'='*50}\n")
        
        return evaluation
    
    def save(self, filepath):
        """Modeli kaydet"""
        
        if not self.trained:
            raise ValueError("Henüz eğitilmemiş model kaydedilemez!")
        
        joblib.dump(self.model, filepath)
        logger.info(f"✅ Model kaydedildi: {filepath}")
    
    def load(self, filepath):
        """Modeli yükle"""
        
        self.model = joblib.load(filepath)
        self.trained = True
        logger.info(f"✅ Model yüklendi: {filepath}")
