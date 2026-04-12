"""
Random Forest Training Pipeline
Complete predictive maintenance system with PostgreSQL integration and KPI tracking
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
    """
    Random Forest training and evaluation pipeline for predictive maintenance.
    Implements 6-step orchestration process with PostgreSQL data integration.
    """
    
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
        Load and prepare training data from PostgreSQL database.
        Database connectivity is mandatory - no synthetic data fallback.
        
        Parameters:
            db_table (str): Table name to load from database
            db_query (str): Custom SQL query (alternative to table name)
        
        Raises:
            ValueError: If neither db_table nor db_query provided
            RuntimeError: If PostgreSQL connection fails
        """
        
        logger.info("="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        # Validate inputs
        if not db_table and not db_query:
            logger.error("CRITICAL: Neither db_table nor db_query provided")
            raise ValueError(
                "Database configuration required. Provide either 'db_table' or 'db_query'. "
                "Note: Synthetic data is disabled - database connection is mandatory."
            )
        
        logger.info("Loading data from PostgreSQL database...")
        df = None
        data_file = None
        
        try:
            if db_table:
                logger.info(f"Table: {db_table}")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, table_name=db_table)
            else:
                logger.info("Executing custom SQL query...")
                df = self.data_prep.load_from_database(DATABASE_CONFIG, query=db_query)
            
            if df is None or len(df) == 0:
                raise ValueError("Database query returned empty result set")
            
            logger.info(f"PostgreSQL data loaded successfully: {df.shape[0]} samples")
            
            # Create audit trail snapshot of loaded data
            data_file = DATA_DIR / 'database_snapshot.csv'
            df.to_csv(data_file, index=False)
            logger.info(f"Data snapshot saved for audit: {data_file}")
                
        except Exception as e:
            logger.error(f"CRITICAL DATABASE ERROR: {str(e)}")
            logger.error("Synthetic data fallback is disabled per project policy")
            logger.error("Please verify database connection in src/config.py")
            raise RuntimeError(
                f"Failed to load data from PostgreSQL database. "
                f"Synthetic data fallback is disabled. "
                f"Please verify database connection. Original error: {str(e)}"
            )
        
        # Log data statistics
        logger.info(f"\nData Information:")
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"\nTarget Distribution:")
        target_col = DATA_CONFIG['target_column']
        if target_col in df.columns:
            logger.info(df[target_col].value_counts().to_string())
        
        # Prepare data splits (train/validation/test)
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
        """
        Train Random Forest classifier on prepared data.
        Uses 1000 trees with entropy criterion for information gain.
        """
        
        logger.info("\n" + "="*60)
        logger.info("STEP 2: RANDOM FOREST MODEL TRAINING")
        logger.info("="*60)
        
        self.model.train(self.X_train, self.y_train)
        
    def evaluate_model(self):
        """
        Evaluate model performance on test set.
        Computes accuracy, precision, recall, F1-score, and feature importance.
        """
        
        logger.info("\n" + "="*60)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*60)
        
        # Evaluate on test set
        self.results['evaluation'] = self.model.evaluate(self.X_test, self.y_test)
        
        # Calculate feature importance ranking
        importance_df = self.model.get_feature_importance(self.feature_names)
        self.results['feature_importance'] = importance_df
        
        logger.info("\nTop 10 Important Features:")
        logger.info(importance_df.head(10).to_string(index=False))
        
    def calculate_kpis(self):
        """
        Calculate model performance KPIs.
        Computes accuracy, precision, recall against target thresholds.
        """
        
        logger.info("\n" + "="*60)
        logger.info("STEP 4: KPI CALCULATION")
        logger.info("="*60)
        
        # Generate predictions on test set
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate model performance metrics
        model_kpis = self.kpi_metrics.calculate_model_performance_kpis(
            self.y_test, y_pred, y_pred_proba
        )
        
        self.results['model_kpis'] = model_kpis
        logger.info("Model performance KPIs calculated")
        logger.info(f"   Accuracy: {model_kpis.get('accuracy', 'N/A')}")
        logger.info(f"   Precision: {model_kpis.get('precision', 'N/A')}")
        logger.info(f"   Recall: {model_kpis.get('recall', 'N/A')}")
        
    def save_model(self):
        """
        Persist trained model and metadata to disk.
        Saves both model weights and feature names for inference.
        """
        
        logger.info("\n" + "="*60)
        logger.info("STEP 5: MODEL PERSISTENCE")
        logger.info("="*60)
        
        filepath = MODELS_DIR / 'random_forest_model.pkl'
        self.model.save(filepath)
        
        # Save feature names for proper column ordering during inference
        np.save(MODELS_DIR / 'feature_names.npy', self.feature_names)
        logger.info(f"Feature names saved: {MODELS_DIR / 'feature_names.npy'}")
        
    def generate_report(self):
        """
        Generate final training report with metrics summary.
        Writes results to CSV file for record-keeping.
        """
        
        logger.info("\n" + "="*60)
        logger.info("STEP 6: REPORT GENERATION")
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
        logger.info("RANDOM FOREST - TRAINING SUMMARY")
        logger.info("-"*60)
        logger.info(f"Accuracy:         {report['accuracy']:.4f}")
        logger.info(f"Precision:        {report['precision']:.4f}")
        logger.info(f"Recall:           {report['recall']:.4f}")
        logger.info(f"F1-Score:         {report['f1_score']:.4f}")
        logger.info(f"Training Samples: {report['training_samples']}")
        logger.info(f"Test Samples:     {report['test_samples']}")
        logger.info(f"Features:         {report['features']}")
        logger.info("-"*60)
        
        logger.info("\nTraining artifacts saved:")
        logger.info(f"   Models:  {MODELS_DIR}")
        logger.info(f"   Logs:    {LOGS_DIR}")
        
    def run_pipeline(self, db_table=None, db_query=None):
        """
        Execute complete training pipeline.
        Orchestrates 6-step process: data prep -> training -> evaluation -> KPI -> save -> report
        
        Parameters:
            db_table (str): Database table to load from
            db_query (str): Custom SQL query
        
        Returns:
            bool: True if pipeline succeeded, False otherwise
        """
        
        logger.info("\n" + "="*80)
        logger.info("RANDOM FOREST PREDICTIVE MAINTENANCE SYSTEM (DATABASE MODE)")
        logger.info("="*80)
        
        try:
            # Step 1: Data Preparation
            self.prepare_data(db_table=db_table, db_query=db_query)
            
            # Step 2: Model Training
            self.train_model()
            
            # Step 3: Model Evaluation
            self.evaluate_model()
            
            # Step 4: KPI Calculation
            self.calculate_kpis()
            
            # Step 5: Model Persistence
            self.save_model()
            
            # Step 6: Report Generation
            self.generate_report()
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            return False


def main():
    """
    Entry point for Random Forest training  system.
    
    Database connection is mandatory for all operations.
    Synthetic data fallback is disabled per system policy.
    """
    
    pipeline = RandomForestPipeline()
    
    # Load from database table
    success = pipeline.run_pipeline(
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
