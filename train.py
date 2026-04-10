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
        
    def prepare_data(self, data_source='synthetic', db_table=None, db_query=None):
        """
        Load and prepare data
        
        Parameters:
        - data_source: 'synthetic', 'csv', or 'postgresql'
        - db_table: Table name (for postgresql)
        - db_query: Custom SQL query (for postgresql)
        """
        
        logger.info("="*60)
        logger.info("ADIM 1: VERİ HAZIRLAMASI")
        logger.info("="*60)
        
        df = None
        data_file = None
        
        if data_source == 'synthetic':
            logger.info("📊 Creating synthetic data...")
            df = create_synthetic_data(n_samples=2000)
            data_file = DATA_DIR / 'synthetic_maintenance_data.csv'
            df.to_csv(data_file, index=False)
            logger.info(f"✅ Synthetic data created: {df.shape[0]} samples")
            
        elif data_source == 'csv':
            # Load from existing CSV file
            data_file = DATA_DIR / 'maintenance_data.csv'
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
            df = pd.read_csv(data_file)
            logger.info(f"✅ CSV data loaded: {df.shape[0]} samples")
            
        elif data_source == 'postgresql':
            logger.info("🔌 Loading data from PostgreSQL...")
            
            if not db_table and not db_query:
                raise ValueError("For postgresql, provide either 'db_table' or 'db_query'")
            
            try:
                if db_table:
                    logger.info(f"📋 Table: {db_table}")
                    df = self.data_prep.load_from_database(DATABASE_CONFIG, table_name=db_table)
                else:
                    logger.info(f"🔍 Executing custom SQL query...")
                    df = self.data_prep.load_from_database(DATABASE_CONFIG, query=db_query)
                
                logger.info(f"✅ PostgreSQL data loaded: {df.shape[0]} samples")
                
                # Save to CSV for future reference
                data_file = DATA_DIR / 'postgresql_maintenance_data.csv'
                df.to_csv(data_file, index=False)
                logger.info(f"💾 Data saved to: {data_file}")
                
            except Exception as e:
                logger.error(f"❌ PostgreSQL connection failed: {e}")
                logger.info("⚠️  Falling back to synthetic data...")
                df = create_synthetic_data(n_samples=2000)
                data_file = DATA_DIR / 'synthetic_maintenance_data.csv'
                df.to_csv(data_file, index=False)
        
        else:
            raise ValueError(f"Unknown data_source: {data_source}")
        
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
        """KPI'ları hesapla"""
        
        logger.info("\n" + "="*60)
        logger.info("ADIM 4: KPI HESAPLAMASı")
        logger.info("="*60)
        
        # Test setinden tahminler al
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Model Performance KPI'ları
        model_kpis = self.kpi_metrics.calculate_model_performance_kpis(
            self.y_test, y_pred, y_pred_proba
        )
        
        # Operational KPI'ları (simüle edilmiş)
        # Create boolean mask safely for numpy arrays or series
        y_test_vals = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        failure_mask = (y_test_vals == 1)
        
        failure_data = pd.DataFrame({
            'failure_time': pd.date_range('2024-01-01', periods=len(self.y_test), freq='H')
        })
        failure_data = failure_data[failure_mask].reset_index(drop=True)
        
        error_data = pd.DataFrame({
            'error_id': range(len(self.y_test)),
            'severity': ['critical' if random_val > 0.8 else 'warning' 
                        for random_val in np.random.random(len(self.y_test))]
        })
        
        operational_kpis = self.kpi_metrics.calculate_operational_kpis(
            failure_data, error_data
        )
        
        # System KPI'ları (simüle edilmiş)
        inference_times = np.random.normal(loc=5, scale=2, size=100) / 1000
        uptime_data = {
            'uptime_percentage': 0.9995,
            'connectivity_success_rate': 0.9720
        }
        
        system_kpis = self.kpi_metrics.calculate_system_kpis(
            inference_times, uptime_data, len(error_data)
        )
        
        # Mali KPI'ları
        avoided_failures = int(np.sum(y_pred) * 0.8)
        
        financial_kpis = self.kpi_metrics.calculate_financial_kpis(
            avoided_failures=avoided_failures,
            baseline_failures=int(len(self.y_test) * 0.15),
            cost_per_failure=FINANCIAL_CONFIG['cost_per_failure'],
            system_cost=FINANCIAL_CONFIG['system_cost'],
            avoided_maintenance_cost=avoided_failures * FINANCIAL_CONFIG['maintenance_cost_per_robot']
        )
        
        # Kapsamlı rapor oluştur
        kpi_report = self.kpi_metrics.generate_kpi_report(
            model_kpis, operational_kpis, system_kpis, financial_kpis
        )
        
        self.results['kpi_report'] = kpi_report
        
        # Özet göster
        self.kpi_metrics.display_kpi_summary(kpi_report)
        
        # CSV'ye kaydet
        report_df = pd.DataFrame([{
            **{f'model_{k}': v for k, v in model_kpis.items()},
            **{f'operational_{k}': v for k, v in operational_kpis.items()},
            **{f'system_{k}': v for k, v in system_kpis.items()},
            **{f'financial_{k}': v for k, v in financial_kpis.items()},
        }])
        report_df.to_csv(LOGS_DIR / 'kpi_report.csv', index=False)
        logger.info(f"✅ KPI raporu kaydedildi: {LOGS_DIR / 'kpi_report.csv'}")
        
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
        
    def run_pipeline(self, data_source='synthetic', db_table=None, db_query=None):
        """
        Pipeline'ı çalıştır
        
        Parameters:
        - data_source: 'synthetic', 'csv', or 'postgresql'
        - db_table: Table name (for postgresql)
        - db_query: Custom SQL query (for postgresql)
        """
        
        logger.info("\n" + "="*80)
        logger.info("RANDOM FOREST ARIZA TAHMIN SİSTEMİ")
        logger.info("="*80)
        
        try:
            # Adım 1: Veri Hazırlama
            self.prepare_data(data_source=data_source, db_table=db_table, db_query=db_query)
            
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
    """Ana girişi - Veri kaynağı seçimi ile"""
    
    # Default: Sentetik veri
    # Eğer PostgreSQL'i kullanmak istirsen:
    pipeline = RandomForestPipeline()
    
    # ========== VERİ KAYNAĞI SEÇİMİ ==========
    # 1. SYNTHETTIK VERİ (Default)
    #success = pipeline.run_pipeline(data_source='synthetic')
    
    # 2. PostgreSQL'den VERİ
    success = pipeline.run_pipeline(
        data_source='postgresql',
        db_table='robots_data'  # Tablo adını değiştir
 )
    
    # 3. PostgreSQL'den CUSTOM SORGU
    # success = pipeline.run_pipeline(
    #     data_source='postgresql',
    #     db_query="""
    #     SELECT * FROM robots_data 
    #     WHERE failure IN (0, 1)
    #     """
    # )
    
    # 4. CSV DÖSYASı
    # success = pipeline.run_pipeline(data_source='csv')
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
