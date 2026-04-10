"""
Quick Start Guide - Example Usage of Predictive Maintenance System

IMPORTANT: Synthetic data is DISABLED. All examples use PostgreSQL database.
Database connection is required to run these examples.

This file demonstrates how to use different components of the system with real database data.
Run sections individually to explore the system.
"""

# Database Configuration
DATABASE_CONFIG = {
    'type': 'postgresql',
    'host': '149.102.155.77',
    'port': 5433,
    'database': 'robot_pipeline',
    'user': 'robot_pipeline_admin',
    'password': 'RobotPipe!2026#PG!149',
    'ssl_mode': 'disable',
}

# ============================================================================
# EXAMPLE 1: Load Data from PostgreSQL Database
# ============================================================================

def example_1_load_database():
    """Load and prepare data from PostgreSQL (Database Required)"""
    
    from src.data_preparation import DataPreparation
    import pandas as pd
    
    print("Loading data from PostgreSQL database...")
    data_prep = DataPreparation(random_state=42)
    
    try:
        # Load from database table
        df = data_prep.load_from_database(
            db_config=DATABASE_CONFIG,
            table_name='robots_data'  # Change table name as needed
        )
        
        print(f"✅ Data loaded successfully: {df.shape[0]} samples")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nTarget distribution:\n{df['failure'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load data from database: {str(e)}")
        print("Please check database configuration in DATABASE_CONFIG")
        raise


# ============================================================================
# EXAMPLE 2: Prepare Data with Validation Set
# ============================================================================

def example_2_prepare_data():
    """Prepare data from database with train/validation/test split (70/15/15)"""
    
    from src.data_preparation import DataPreparation
    
    print("Loading and preparing data from PostgreSQL...")
    data_prep = DataPreparation(random_state=42)
    
    try:
        # Load from database
        df = data_prep.load_from_database(
            db_config=DATABASE_CONFIG,
            table_name='robots_data'
        )
        
        # Save to temporary CSV for preprocessing
        df.to_csv('data/temp_db_data.csv', index=False)
        
        # Prepare data with validation set
        X_train, X_val, X_test, y_train, y_val, y_test, features = data_prep.prepare_data(
            filepath='data/temp_db_data.csv',
            target_column='failure',
            numerical_cols=['temperature', 'vibration', 'pressure', 'humidity',
                           'operational_hours', 'error_count', 'last_maintenance_days',
                           'robot_age_months', 'power_consumption'],
            validation_size=0.15,
            return_validation=True
        )
        
        print(f"✅ Data preparation complete")
        print(f"   Training set: {X_train.shape[0]} samples (70%)")
        print(f"   Validation set: {X_val.shape[0]} samples (15%)")
        print(f"   Test set: {X_test.shape[0]} samples (15%)")
        print(f"   Features: {len(features)}")
    from src.tree_models import TreeBasedModels
    from src.data_preparation import create_synthetic_data, DataPreparation
    import pandas as pd
    import numpy as np
    
    # Prepare and train data
    df = create_synthetic_data(n_samples=1000)
    df.to_csv('data/temp_data.csv', index=False)
    
    data_prep = DataPreparation()
    X_train, X_test, y_train, y_test, features = data_prep.prepare_data(
        filepath='data/temp_data.csv',
        target_column='failure',
        numerical_cols=['temperature', 'vibration', 'pressure', 'humidity',
                       'operational_hours', 'error_count', 'last_maintenance_days',
                       'robot_age_months', 'power_consumption'],
        test_size=0.2
    )
    
    # Train model
    models = TreeBasedModels()
    models.train_model('random_forest', X_train, y_train)
    
    # Get predictions
    y_pred = models.predict('random_forest', X_test)
    y_pred_proba = models.predict_proba('random_forest', X_test)
    
    # Calculate KPIs
    print("Calculating KPIs...")
    kpi = KPIMetrics()
    
    # Model Performance KPIs
    model_kpis = kpi.calculate_model_performance_kpis(y_test, y_pred, y_pred_proba)
    
    print("\n=== MODEL PERFORMANCE KPIs ===")
    print(f"Accuracy: {model_kpis['prediction_accuracy']:.4f}")
    print(f"Recall: {model_kpis['recall']:.4f}")
    print(f"Precision: {model_kpis['precision']:.4f}")
    print(f"F1-Score: {model_kpis['f1_score']:.4f}")
    print(f"False Alarm Rate: {model_kpis['false_alarm_rate']:.4f}")
    
    # Operational KPIs
    failure_data = pd.DataFrame({
        'failure_time': pd.date_range('2024-01-01', periods=len(y_test), freq='H')
    }).loc[y_test == 1]
    
    error_data = pd.DataFrame({
        'error_id': range(len(y_test)),
        'severity': ['critical' if np.random.random() > 0.8 else 'warning' 
                    for _ in range(len(y_test))]
    })
    
    operational_kpis = kpi.calculate_operational_kpis(failure_data, error_data)
    
    print("\n=== OPERATIONAL KPIs ===")
    print(f"Error Rate: {operational_kpis['error_rate']:.4f}")
    print(f"Critical Error Rate: {operational_kpis['critical_error_rate']:.4f}")
    
    # System KPIs
    inference_times = np.random.normal(loc=5, scale=2, size=100) / 1000  # ms to s
    system_kpis = kpi.calculate_system_kpis(
        inference_times, 
        {'uptime_percentage': 0.9995, 'connectivity_success_rate': 0.972},
        len(error_data)
    )
    
    print("\n=== SYSTEM KPIs ===")
    print(f"System Latency: {system_kpis['system_latency']:.4f}s")
    print(f"System Uptime: {system_kpis['system_uptime']:.4f}")
    
    # Financial KPIs
    financial_kpis = kpi.calculate_financial_kpis(
        avoided_failures=25,
        baseline_failures=100,
        cost_per_failure=50000,
        system_cost=500000,
        avoided_maintenance_cost=250000
    )
    
    print("\n=== FINANCIAL KPIs ===")
    print(f"Avoided Failures: {financial_kpis['avoided_failures']}")
    print(f"Cost Savings: ${financial_kpis['cost_savings']:,.2f}")
    print(f"ROI: {financial_kpis['roi']:.2%}")
    
    return model_kpis, operational_kpis, system_kpis, financial_kpis


# ============================================================================
# EXAMPLE 4: Feature Importance Analysis
# ============================================================================

def example_4_feature_importance():
    """Analyze feature importance of trained models"""
    
    from src.tree_models import TreeBasedModels
    from src.data_preparation import create_synthetic_data, DataPreparation
    import pandas as pd
    
    # Prepare data
    df = create_synthetic_data(n_samples=1000)
    df.to_csv('data/temp_data.csv', index=False)
    
    data_prep = DataPreparation()
    X_train, X_test, y_train, y_test, features = data_prep.prepare_data(
        filepath='data/temp_data.csv',
        target_column='failure',
        numerical_cols=['temperature', 'vibration', 'pressure', 'humidity',
                       'operational_hours', 'error_count', 'last_maintenance_days',
                       'robot_age_months', 'power_consumption'],
        test_size=0.2
    )
    
    # Train model
    print("Training Random Forest for feature importance analysis...")
    models = TreeBasedModels()
    models.train_model('random_forest', X_train, y_train)
    
    # Get feature importance
    importance = models.get_feature_importance('random_forest')
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Random Forest):")
    print(importance_df.to_string(index=False))
    
    # Plot top features
    print("\nTop 5 Most Important Features:")
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    return importance_df


# ============================================================================
# EXAMPLE 5: Make Predictions
# ============================================================================

def example_5_predictions():
    """Make predictions on new data"""
    
    import pandas as pd
    import numpy as np
    from src.tree_models import TreeBasedModels
    from src.data_preparation import DataPreparation, create_synthetic_data
    
    # Prepare and train
    df = create_synthetic_data(n_samples=1000)
    df.to_csv('data/temp_data.csv', index=False)
    
    data_prep = DataPreparation()
    X_train, X_test, y_train, y_test, features = data_prep.prepare_data(
        filepath='data/temp_data.csv',
        target_column='failure',
        numerical_cols=['temperature', 'vibration', 'pressure', 'humidity',
                       'operational_hours', 'error_count', 'last_maintenance_days',
                       'robot_age_months', 'power_consumption'],
        test_size=0.2
    )
    
    # Train model
    models = TreeBasedModels()
    models.train_model('random_forest', X_train, y_train)
    
    # Create sample data
    print("Creating sample data for predictions...")
    sample_data = pd.DataFrame({
        'temperature': [75, 85, 95],
        'vibration': [0.4, 0.6, 0.9],
        'pressure': [95, 110, 125],
        'humidity': [40, 50, 60],
        'operational_hours': [1000, 5000, 8000],
        'error_count': [2, 5, 12],
        'last_maintenance_days': [100, 250, 350],
        'robot_age_months': [12, 60, 90],
        'power_consumption': [450, 550, 650],
    })
    
    # Scale the data (important!)
    sample_data_scaled = data_prep.scaler.transform(sample_data)
    
    # Make predictions
    predictions = models.predict('random_forest', sample_data_scaled)
    probabilities = models.predict_proba('random_forest', sample_data_scaled)
    
    # Display results
    print("\nPredictions:")
    results = pd.DataFrame({
        'sample': ['Robot 1', 'Robot 2', 'Robot 3'],
        'prediction': ['Failure' if p == 1 else 'Normal' for p in predictions],
        'failure_probability': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1)
    })
    
    print(results.to_string(index=False))
    
    return results


# ============================================================================
# MAIN - Run examples
# ============================================================================

if __name__ == '__main__':
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  PREDICTIVE MAINTENANCE - QUICK START EXAMPLES             ║")
    print("╚════════════════════════════════════════════════════════════╝\n")
    
    try:
        # Uncomment the example you want to run:
        
        # print("EXAMPLE 1: Data Preparation")
        # print("="*60)
        # X_train, X_test, y_train, y_test, features = example_1_data_preparation()
        
        # print("\n\nEXAMPLE 2: Train Models")
        # print("="*60)
        # models, X_test, y_test, features = example_2_train_models()
        
        # print("\n\nEXAMPLE 3: Calculate KPIs")
        # print("="*60)
        # model_kpis, operational_kpis, system_kpis, financial_kpis = example_3_calculate_kpis()
        
        # print("\n\nEXAMPLE 4: Feature Importance")
        # print("="*60)
        # importance_df = example_4_feature_importance()
        
        # print("\n\nEXAMPLE 5: Make Predictions")
        # print("="*60)
        # results = example_5_predictions()
        
        print("All examples ready to run!")
        print("\nTo run an example, uncomment the desired example in the main section.")
        print("\nAvailable examples:")
        print("  1. example_1_data_preparation()   - Load and prepare data")
        print("  2. example_2_train_models()       - Train tree-based models")
        print("  3. example_3_calculate_kpis()     - Calculate performance KPIs")
        print("  4. example_4_feature_importance() - Analyze feature importance")
        print("  5. example_5_predictions()        - Make predictions on new data")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nNote: Make sure to run the full pipeline first: python train.py")
