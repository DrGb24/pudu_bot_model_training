#!/usr/bin/env python3
"""
LSTM Training Pipeline for Predictive Maintenance
TensorFlow/Keras based time-series model training
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG, DATA_CONFIG, MODEL_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from kpi_metrics import KPIMetrics

# Setup logging
LOG_DIR = Path(__file__).parent / 'logs' / 'lstm'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / 'data'
MODEL_DIR = Path(__file__).parent / 'models' / 'lstm'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class LSTMTrainingPipeline:
    """6-step LSTM training orchestration"""
    
    def __init__(self):
        self.data_prep = DataPreparation()
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.results = {}
        self.feature_names = None
        
    def prepare_data(self, db_table: str = None, db_query: str = None, sequence_length: int = 10):
        """ADIM 1: VERİ HAZIRLAMASI (LSTM için zaman serisi hazırlama)"""
        
        logger.info("="*60)
        logger.info("ADIM 1: LSTM VERİ HAZIRLAMASI (TIME-SERIES)")
        logger.info("="*60)
        
        # Validate input
        if not db_table and not db_query:
            logger.error("❌ CRITICAL: Neither db_table nor db_query provided")
            raise ValueError("Database configuration required!")
        
        logger.info("🔌 Loading data from PostgreSQL database...")
        
        try:
            if db_table:
                logger.info(f"📋 Table: {db_table}")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, table_name=db_table)
            else:
                logger.info(f"🔍 Executing custom SQL query...")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, query=db_query)
            
            if df is None or len(df) == 0:
                raise ValueError("Database query returned empty result set")
            
            logger.info(f"✅ PostgreSQL data loaded: {df.shape[0]} samples")
            
            # Save snapshot
            data_file = DATA_DIR / 'lstm_database_snapshot.csv'
            df.to_csv(data_file, index=False)
            logger.info(f"💾 Snapshot saved: {data_file}")
            
            # Show data info
            logger.info(f"\n📊 Data Info:")
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Columns: {list(df.columns)}")
            
            # Create time-series sequences
            logger.info(f"\n⏱️  Creating sequences (length={sequence_length})...")
            X, y = self._create_sequences(
                df,
                target_col=DATA_CONFIG['target_column'],
                sequence_length=sequence_length
            )
            
            logger.info(f"✅ Sequences created: {X.shape}")
            
            # Train/val/test split
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            self.X_train = X[:train_size]
            self.X_val = X[train_size:train_size+val_size]
            self.X_test = X[train_size+val_size:]
            
            self.y_train = y[:train_size]
            self.y_val = y[train_size:train_size+val_size]
            self.y_test = y[train_size+val_size:]
            
            logger.info(f"Training set:   {self.X_train.shape}")
            logger.info(f"Validation set: {self.X_val.shape}")
            logger.info(f"Test set:       {self.X_test.shape}")
            
            # Feature names from original dataframe
            self.feature_names = [col for col in df.columns if col != DATA_CONFIG['target_column']]
            logger.info(f"Features: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"❌ Database error: {str(e)}")
            raise RuntimeError(f"Failed to load data: {str(e)}")
    
    def _create_sequences(self, df: pd.DataFrame, target_col: str, sequence_length: int = 10):
        """Create time-series sequences from dataframe"""
        
        from sklearn.preprocessing import StandardScaler
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X_data = df[feature_cols].values
        y_data = df[target_col].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_data)
        
        # Save scaler
        import joblib
        scaler_path = MODEL_DIR / 'lstm_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"💾 Scaler saved: {scaler_path}")
        
        # Create sequences
        X_seq = []
        y_seq = []
        
        for i in range(len(X_scaled) - sequence_length):
            X_seq.append(X_scaled[i:i+sequence_length])
            y_seq.append(y_data[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self, lstm_units: int = 128, dropout_rate: float = 0.2):
        """ADIM 2: LSTM MODELİ YAPISI"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 2: LSTM MODELİ YAPISI")
        logger.info("="*60)
        
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        self.model = LSTMModel(
            input_shape=input_shape,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=64,
            learning_rate=0.001
        )
        
        self.model.build_model()
        self.model.model.summary()
    
    def train_model(self, epochs: int = 100, batch_size: int = 32):
        """ADIM 3: LSTM MODELİ EĞİTİMİ"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 3: LSTM MODELİ EĞİTİMİ")
        logger.info("="*60)
        
        history = self.model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Save history
        import json
        history_path = LOG_DIR / 'training_history.json'
        
        # Convert to serializable format
        history_dict = {}
        for key, values in history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        logger.info(f"💾 History saved: {history_path}")
    
    def evaluate_model(self):
        """ADIM 4: MODEL DEĞERLENDİRMESİ"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 4: MODEL DEĞERLENDİRMESİ")
        logger.info("="*60)
        
        self.results['evaluation'] = self.model.evaluate(self.X_test, self.y_test)
    
    def calculate_kpis(self):
        """ADIM 5: KPI HESAPLAMASı"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 5: KPI HESAPLAMASı (LSTM)")
        logger.info("="*60)
        
        kpi_calculator = KPIMetrics()
        
        y_pred = (self.model.predict(self.X_test) > 0.5).astype(int).flatten()
        
        kpi_results = kpi_calculator.calculate(
            y_test=self.y_test,
            y_pred=y_pred,
            y_scores=self.model.predict(self.X_test).flatten()
        )
        
        self.results['kpi'] = kpi_results
        
        logger.info(f"✅ KPIs calculated")
    
    def save_model(self):
        """ADIM 6: MODEL KAYDETME"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 6: MODEL KAYDETME")
        logger.info("="*60)
        
        model_path = MODEL_DIR / 'lstm_model.h5'
        self.model.save_model(str(model_path))
        
        # Save feature names
        import numpy as np
        features_path = MODEL_DIR / 'lstm_feature_names.npy'
        np.save(features_path, self.feature_names)
        logger.info(f"✅ Features saved: {features_path}")
    
    def generate_report(self):
        """ADIM 7: RAPOR OLUŞTURMA"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 7: RAPOR OLUŞTURMA")
        logger.info("="*60)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'model': ['LSTM'],
            'accuracy': [self.results['evaluation'].get('accuracy', 0)],
            'precision': [self.results['evaluation'].get('precision', 0)],
            'recall': [self.results['evaluation'].get('recall', 0)],
            'f1_score': [self.results['evaluation'].get('f1_score', 0)],
            'training_samples': [len(self.X_train)],
            'test_samples': [len(self.X_test)],
            'sequence_length': [self.X_train.shape[1]],
            'features': [self.X_train.shape[2]]
        })
        
        report_path = LOG_DIR / 'lstm_final_report.csv'
        results_df.to_csv(report_path, index=False)
        logger.info(f"✅ Report saved: {report_path}")
        
        # Print summary
        logger.info("\n" + "-"*60)
        logger.info("LSTM MODEL - ÖZET RAPOR")
        logger.info("-"*60)
        logger.info(f"Doğruluk:      {self.results['evaluation']['accuracy']:.4f} ✅")
        logger.info(f"Kesinlik:      {self.results['evaluation']['precision']:.4f} ✅")
        logger.info(f"Hatırlanma:    {self.results['evaluation']['recall']:.4f}")
        logger.info(f"F1-Score:      {self.results['evaluation']['f1_score']:.4f}")
        logger.info(f"Eğitim Verisi: {len(self.X_train)} örnek")
        logger.info(f"Test Verisi:   {len(self.X_test)} örnek")
        logger.info(f"Seq. Length:   {self.X_train.shape[1]}")
        logger.info(f"Features:      {self.X_train.shape[2]}")
        logger.info("-"*60)
        
        logger.info(f"\n✅ Sonuçlar kaydedildi:")
        logger.info(f"   Modeller: {MODEL_DIR}")
        logger.info(f"   Loglar:   {LOG_DIR}")
    
    def run_pipeline(self, db_table: str = None, db_query: str = None, 
                     sequence_length: int = 10, epochs: int = 100):
        """Run complete 7-step pipeline"""
        
        logger.info("\n" + "="*80)
        logger.info("LSTM ARIZA TAHMIN SİSTEMİ (DATABASE MODE)")
        logger.info("="*80 + "\n")
        
        try:
            self.prepare_data(db_table=db_table, db_query=db_query, sequence_length=sequence_length)
            self.build_model(lstm_units=128, dropout_rate=0.2)
            self.train_model(epochs=epochs, batch_size=32)
            self.evaluate_model()
            self.calculate_kpis()
            self.save_model()
            self.generate_report()
            
            logger.info("\n" + "="*80)
            logger.info("✅ LSTM PIPELINE BAŞARILI ŞEKILDE TAMAMLANDI!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline hatası: {str(e)}")
            raise


def main():
    """Execute LSTM training pipeline"""
    
    pipeline = LSTMTrainingPipeline()
    
    # Use same SQL query as Random Forest but create time-series sequences
    sql_query = """
    SELECT 
      (1 - is_success::int) as failure,
      check_result_count as error_count,
      EXTRACT(HOUR FROM task_time)::int as task_hour,
      EXTRACT(DAY FROM task_time)::int as task_day_of_month,
      EXTRACT(DOW FROM task_time)::int as task_day_of_week,
      LENGTH(robot_id)::int as robot_id_length,
      LENGTH(soft_version)::int as software_version_length,
      CASE WHEN product_code LIKE '%PuduBot%' THEN 1
           WHEN product_code LIKE '%KettyBot%' THEN 2
           WHEN product_code LIKE '%Bellabot%' THEN 3
           WHEN product_code LIKE '%CC%' THEN 4
           ELSE 5 END as product_code_type,
      check_result_count as error_severity,
      COALESCE((SELECT COUNT(*) FROM robot_logs_error e 
               WHERE e.robot_id = robot_logs_info.robot_id 
               AND EXTRACT(HOUR FROM e.task_time) = EXTRACT(HOUR FROM robot_logs_info.task_time)
               AND DATE(e.task_time) = DATE(robot_logs_info.task_time))::float, 0) as hourly_error_rate
    FROM robot_logs_info
    WHERE check_result_count > 0
    LIMIT 2000
    """
    
    success = pipeline.run_pipeline(
        db_query=sql_query,
        sequence_length=10,
        epochs=50
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
