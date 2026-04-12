"""
Random Forest Inference Engine
Loads trained Random Forest model for failure prediction on new robot data
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
    """
    Random Forest-based inference engine for robot failure prediction.
    Provides fast baseline predictions using ensemble tree methods.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.load_model()
        self.load_scaler()
        
    def load_model(self):
        """
        Load pre-trained Random Forest model and feature names from disk.
        
        Raises:
            FileNotFoundError: If model file not found
        """
        model_path = MODELS_DIR / 'random_forest_model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        # Load feature names for proper column ordering
        features_path = MODELS_DIR / 'feature_names.npy'
        if features_path.exists():
            self.feature_names = np.load(features_path, allow_pickle=True)
        
        logger.info("Model loaded successfully")
        logger.info(f"   Features: {len(self.feature_names) if self.feature_names is not None else 'Unknown'}")
        
    def load_scaler(self):
        """
        Load feature normalization scaler from data preparation module.
        Optional - used for input normalization if available.
        """
        try:
            from src.data_preparation import DataPreparation
            self.scaler = DataPreparation().scaler
        except:
            self.scaler = None
    
    def predict(self, X):
        """
        Generate predictions on input samples.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix with robot sensor data
        
        Returns:
            tuple: (predictions, probabilities)
                - predictions: Binary array (0=normal, 1=failure)
                - probabilities: 2D array of class probabilities
        """
        if isinstance(X, pd.DataFrame):
            # Ensure correct column ordering
            if self.feature_names is not None:
                X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def predict_failure_risk(self, X):
        """
        Generate failure risk assessment for robot fleet.
        
        Produces structured output with probabilities and risk categorization.
        
        Args:
            X (pd.DataFrame): Feature data for multiple robots
        
        Returns:
            pd.DataFrame: Results with columns:
                - 'prediction': Categorical prediction (Failure/Normal)
                - 'no_failure_prob': Probability of normal operation
                - 'failure_prob': Probability of failure
                - 'risk_score': Numeric risk score (0-1)
                - 'risk_level': Categorical risk level (LOW/MEDIUM/HIGH)
        """
        predictions, probabilities = self.predict(X)
        
        # Construct results dataframe with risk assessment
        results = pd.DataFrame({
            'prediction': ['Failure' if p == 1 else 'Normal' for p in predictions],
            'no_failure_prob': probabilities[:, 0],
            'failure_prob': probabilities[:, 1],
            'risk_score': probabilities[:, 1],
            'risk_level': pd.cut(probabilities[:, 1], 
                                bins=[0, 0.3, 0.6, 1.0],
                                labels=['LOW', 'MEDIUM', 'HIGH'])
        })
        
        return results


def example_inference():
    """
    Execute inference demonstration with example robot telemetry data.
    Validates that Random Forest model is functional.
    """
    logger.info("="*60)
    logger.info("RANDOM FOREST INFERENCE DEMONSTRATION")
    logger.info("="*60)
    
    # Initialize inference engine
    inference = RandomForestInference()
    
    # Example telemetry data (9 features matching training data)
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
    
    logger.info("\nInput Data:")
    logger.info(sample_data.to_string())
    
    # Generate predictions
    try:
        results = inference.predict_failure_risk(sample_data)
        
        logger.info("\nPrediction Results:")
        logger.info(results.to_string())
        
        # Detailed assessment per robot
        logger.info("\nDetailed Analysis:")
        for idx, row in results.iterrows():
            risk = row['risk_level']
            prob = row['failure_prob']
            logger.info(f"   Robot {idx}: {risk} Risk ({prob*100:.1f}% failure probability)")
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")


if __name__ == '__main__':
    try:
        example_inference()
    except FileNotFoundError as e:
        logger.error(f"❌ Hata: {str(e)}")
        logger.info("   Önce modeli eğitmek için: python train.py")
