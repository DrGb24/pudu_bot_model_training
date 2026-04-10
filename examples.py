"""
Quick Start Guide - Example Usage of Predictive Maintenance System

This file demonstrates how to use different components of the system.
Run sections individually to explore the system.
"""

# ============================================================================
# EXAMPLE 1: Basic Data Preparation
# ============================================================================

def example_1_data_preparation():
    """Load and prepare data"""
    
    from src.data_preparation import DataPreparation, create_synthetic_data
    import pandas as pd
    
    # Create synthetic data
    print("Creating synthetic maintenance data...")
    df = create_synthetic_data(n_samples=1000)
    print(f"Created data with shape: {df.shape}")
    print(f"\nTarget distribution:\n{df['failure'].value_counts()}")
    
    # Prepare data
    print("\nPreparing data for training...")
    data_prep = DataPreparation(random_state=42)
    
    X_train, X_test, y_train, y_test, features = data_prep.prepare_data(
        filepath='data/synthetic_maintenance_data.csv',
        target_column='failure',
        numerical_cols=['temperature', 'vibration', 'pressure', 'humidity',
                       'operational_hours', 'error_count', 'last_maintenance_days',
                       'robot_age_months', 'power_consumption'],
        test_size=0.2
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Features: {features}")
    
    return X_train, X_test, y_train, y_test, features


# ============================================================================
# EXAMPLE 2: Train Individual Models
# ============================================================================

def example_2_train_models():
    """Train specific tree-based models"""
    
    from src.tree_models import TreeBasedModels
    from src.data_preparation import create_synthetic_data, DataPreparation
    
    # Prepare data first
    print("Preparing data...")
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
    
    # Train models
    print("\nTraining models...")
    models = TreeBasedModels(random_state=42)
    
    # Train individual models
    model_names = ['random_forest', 'gradient_boosting', 'xgboost']
    
    for model_name in model_names:
        print(f"Training {model_name}...")
        models.train_model(model_name, X_train, y_train, X_test, y_test)
    
    # Compare models
    print("\nComparing models...")
    comparison = models.compare_models(X_test, y_test)
    print(comparison)
    
    # Get best model
    best_model, metrics = models.get_best_model(X_test, y_test)
    print(f"\nBest Model: {best_model}")
    print(f"Metrics:\n{metrics}")
    
    return models, X_test, y_test, features


# ============================================================================
# EXAMPLE 3: Calculate KPIs
# ============================================================================

def example_3_calculate_kpis():
    """Calculate performance KPIs"""
    
    from src.kpi_metrics import KPIMetrics
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
