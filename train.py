"""
Main training and evaluation pipeline
Complete predictive maintenance system with KPI tracking
"""

import numpy as np
import pandas as pd
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preparation import DataPreparation, create_synthetic_data
from tree_models import RandomForestModel
from kpi_metrics import KPIMetrics
from config import (
    DATA_DIR, MODELS_DIR, LOGS_DIR, 
    MODEL_CONFIG, DATA_CONFIG, FINANCIAL_CONFIG, DATABASE_CONFIG
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RandomForestPipeline:
    """Random Forest Arıza Tahmin Pipeline"""
    
    def __init__(self):
        self.data_prep = DataPreparation(random_state=MODEL_CONFIG['random_state'])
        self.model = RandomForestModel(random_state=MODEL_CONFIG['random_state'])
        self.kpi_metrics = KPIMetrics()
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        
    def prepare_data(self, db_table=None, db_query=None):
        """
        Load and prepare data from PostgreSQL database (MANDATORY)
        
        Parameters:
        - db_table: Table name to load from database
        - db_query: Custom SQL query (optional alternative to table name)
        
        Raises:
        - ValueError: If neither db_table nor db_query provided
        - RuntimeError: If PostgreSQL connection fails (no fallback allowed)
        """
        
        logger.info("="*60)
        logger.info("ADIM 1: VERİ HAZIRLAMASI (DATABASE REQUIRED)")
        logger.info("="*60)
        
        # Validate input
        if not db_table and not db_query:
            logger.error("❌ CRITICAL: Neither db_table nor db_query provided")
            raise ValueError(
                "❌ Database configuration required! Provide either 'db_table' or 'db_query'. "
                "Synthetic data is DISABLED - database connection is mandatory."
            )
        
        logger.info("🔌 Loading data from PostgreSQL database...")
        df = None
        data_file = None
        
        try:
            if db_table:
                logger.info(f"📋 Table: {db_table}")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, table_name=db_table)
            else:
                logger.info(f"🔍 Executing custom SQL query...")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, query=db_query)
            
            if df is None or len(df) == 0:
                raise ValueError("Database query returned empty result set")
            
            logger.info(f"✅ PostgreSQL data loaded successfully: {df.shape[0]} samples")
            
            # Archive data snapshot for audit trail
            data_file = DATA_DIR / 'database_snapshot.csv'
            df.to_csv(data_file, index=False)
            logger.info(f"💾 Data snapshot saved for audit: {data_file}")
                
        except Exception as e:
            logger.error(f"❌ CRITICAL DATABASE ERROR: {str(e)}")
            logger.error("❌ Synthetic data fallback is DISABLED per project policy")
            logger.error("⚠️  Database configuration: Please check src/config.py")
            raise RuntimeError(
                f"Failed to load data from PostgreSQL database. "
                f"Synthetic data fallback is disabled. "
                f"Please verify database connection. Original error: {str(e)}"
            )
        
        # Show data info
        logger.info(f"\n📊 Data Info:")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"\n📈 Target Distribution:")
        target_col = DATA_CONFIG['target_column']
        if target_col in df.columns:
            logger.info(df[target_col].value_counts().to_string())
        
        # Prepare data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test, feature_names = \
            self.data_prep.prepare_data(
                filepath=data_file,
                target_column=DATA_CONFIG['target_column'],
                categorical_cols=DATA_CONFIG['categorical_columns'],
                numerical_cols=DATA_CONFIG['numerical_columns'],
                validation_size=MODEL_CONFIG.get('validation_size', 0.15),
                return_validation=True
            )
        
        self.feature_names = feature_names
        logger.info(f"Training set: {self.X_train.shape}")
        logger.info(f"Validation set: {self.X_val.shape}")
        logger.info(f"Test set: {self.X_test.shape}")
        logger.info(f"Features: {len(feature_names)}")
        
    def train_model(self):
        """Random Forest modelini eğit"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 2: RANDOM FOREST MODELİ EĞİTİMİ")
        logger.info("="*60)
        
        self.model.train(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """Modeli test setinde değerlendir"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 3: MODEL DEĞERLENDİRMESİ")
        logger.info("="*60)
        
        # Model değerlendirmesi
        self.results['evaluation'] = self.model.evaluate(self.X_test, self.y_test)
        
        # En önemli özellikleri bul
        importance_df = self.model.get_feature_importance(self.feature_names)
        self.results['feature_importance'] = importance_df
        
        logger.info(f"\n📊 EN ÖNEMLİ 10 ÖZELLİK:")
        logger.info(importance_df.head(10).to_string(index=False))
        
    def calculate_kpis(self):
        """KPI'ları hesapla - Model performance only (skip operational/system KPIs for now)"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 4: KPI HESAPLAMASı (MODEL PERFORMANCE)")
        logger.info("="*60)
        
        # Test setinden tahminler al
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Model Performance KPI'ları ONLY (skip operational/system KPIs to avoid datetime errors)
        model_kpis = self.kpi_metrics.calculate_model_performance_kpis(
            self.y_test, y_pred, y_pred_proba
        )
        
        self.results['model_kpis'] = model_kpis
        logger.info(f"✅ Model KPIs calculated")
        logger.info(f"   - Accuracy: {model_kpis.get('accuracy', 'N/A')}")
        logger.info(f"   - Precision: {model_kpis.get('precision', 'N/A')}")
        logger.info(f"   - Recall: {model_kpis.get('recall', 'N/A')}")
        
    def save_model(self):
        """Modeli kaydet"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 5: MODEL KAYDETME")
        logger.info("="*60)
        
        filepath = MODELS_DIR / 'random_forest_model.pkl'
        self.model.save(filepath)
        
        # Özellik adlarını kaydet
        np.save(MODELS_DIR / 'feature_names.npy', self.feature_names)
        logger.info(f"✅ Özellik adları kaydedildi: {MODELS_DIR / 'feature_names.npy'}")
        
    def generate_report(self):
        """Son raporu oluştur"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 6: RAPOR OLUŞTURMA")
        logger.info("="*60)
        
        eval_results = self.results['evaluation']
        
        report = {
            'model': 'Random Forest',
            'accuracy': eval_results['accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1_score': eval_results['f1_score'],
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features': len(self.feature_names),
        }
        
        report_df = pd.DataFrame([report])
        report_df.to_csv(LOGS_DIR / 'final_report.csv', index=False)
        
        logger.info("\n" + "-"*60)
        logger.info("RANDOM FOREST - ÖZET RAPOR")
        logger.info("-"*60)
        logger.info(f"Doğruluk:      {report['accuracy']:.4f} ✅")
        logger.info(f"Kesinlik:      {report['precision']:.4f} ✅")
        logger.info(f"Hatırlanma:    {report['recall']:.4f} ✅")
        logger.info(f"F1-Score:      {report['f1_score']:.4f} ✅")
        logger.info(f"Eğitim Verisi: {report['training_samples']} örnek")
        logger.info(f"Test Verisi:   {report['test_samples']} örnek")
        logger.info(f"Özellikler:    {report['features']}")
        logger.info("-"*60)
        
        logger.info(f"\n✅ Sonuçlar kaydedildi:")
        logger.info(f"   Modeller: {MODELS_DIR}")
        logger.info(f"   Loglar:   {LOGS_DIR}")
        
    def run_pipeline(self, db_table=None, db_query=None):
        """
        Training Pipeline - PostgreSQL Database Required
        
        Parameters:
        - db_table: Table name to load from database (recommended)
        - db_query: Custom SQL query (optional alternative)
        
        NOTE: Synthetic data is DISABLED. Database connection is MANDATORY.
        """
        
        logger.info("\n" + "="*80)
        logger.info("RANDOM FOREST ARIZA TAHMIN SİSTEMİ (DATABASE MODE)")
        logger.info("="*80)
        
        try:
            # Adım 1: PostgreSQL Veri Hazırlama
            self.prepare_data(db_table=db_table, db_query=db_query)
            
            # Adım 2: Model Eğitimi
            self.train_model()
            
            # Adım 3: Model Değerlendirmesi
            self.evaluate_model()
            
            # Adım 4: KPI Hesaplama
            self.calculate_kpis()
            
            # Adım 5: Model Kaydetme
            self.save_model()
            
            # Adım 6: Rapor Oluşturma
            self.generate_report()
            
            logger.info("\n" + "="*80)
            logger.info("✅ PIPELINE BAŞARILI ŞEKILDE TAMAMLANDI!")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline hatası: {str(e)}", exc_info=True)
            return False


def main():
    """Training Pipeline - PostgreSQL Database Required
    
    NOTE: Synthetic data is DISABLED. All training uses PostgreSQL database only.
    Database connection is MANDATORY for the system to work.
    """
    
    pipeline = RandomForestPipeline()
    
    # ========== DATABASE CONFIGURATION ==========
    # Load from robot_logs_info with is_success flag (inverted to failure)
    # Based on HATA_KODLARI_ROBOT.xlsx severity - errors as actual failure indicators
    success = pipeline.run_pipeline(
        db_query="""
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
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
