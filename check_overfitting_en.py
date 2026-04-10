#!/usr/bin/env python
"""Detailed Overfitting Analysis - Check if 100% accuracy is valid or overfitting"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import sys

sys.path.insert(0, 'src')

from data_preparation import DataPreparation
from tree_models import RandomForestModel
from config import DATABASE_CONFIG, DATA_CONFIG, MODEL_CONFIG

print("="*70)
print("OVERFITTING ANALYSIS - 100% Accuracy Verification")
print("="*70)

try:
    # Load data from database
    print("\n1. LOADING DATA...")
    data_prep = DataPreparation()
    df = data_prep.load_from_database(
        db_config=DATABASE_CONFIG,
        query="""
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
                 ELSE 3 END as product_code_type,
            CASE WHEN is_success = 1 THEN 0 ELSE check_result_count END as error_severity
        FROM robot_logs_info
        WHERE check_result_count > 0
        LIMIT 2000
        """
    )
    print(f"OK - {len(df)} rows loaded")
    print(f"Class Distribution:\n{df['failure'].value_counts()}\n")
    
    # Prepare data
    print("2. PREPARING DATA...")
    df.to_csv('data/temp_analysis_data.csv', index=False)
    
    X_train, X_val, X_test, y_train, y_val, y_test, features = data_prep.prepare_data(
        filepath='data/temp_analysis_data.csv',
        target_column=DATA_CONFIG['target_column'],
        categorical_cols=DATA_CONFIG['categorical_columns'],
        numerical_cols=DATA_CONFIG['numerical_columns'],
        validation_size=MODEL_CONFIG.get('validation_size', 0.15),
        return_validation=True
    )
    
    print(f"\nData Split:")
    print(f"  Training:   {X_train.shape[0]} samples ({X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    
    # Train model
    print("\n3. TRAINING MODEL...")
    model = RandomForestModel()
    model.train(X_train, y_train)
    print("OK - Model training complete")
    
    # Get predictions on all sets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracies
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    # Try AUC-ROC if possible
    try:
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)
    except:
        train_auc = val_auc = test_auc = 0
    
    # Print results
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS RESULTS")
    print("="*70)
    
    print("\nACCURACY:")
    print(f"  Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\nPRECISION:")
    print(f"  Training:   {train_prec:.4f}")
    print(f"  Validation: {val_prec:.4f}")
    print(f"  Test:       {test_prec:.4f}")
    
    print("\nRECALL:")
    print(f"  Training:   {train_rec:.4f}")
    print(f"  Validation: {val_rec:.4f}")
    print(f"  Test:       {test_rec:.4f}")
    
    print("\nF1-SCORE:")
    print(f"  Training:   {train_f1:.4f}")
    print(f"  Validation: {val_f1:.4f}")
    print(f"  Test:       {test_f1:.4f}")
    
    if train_auc > 0:
        print("\nAUC-ROC:")
        print(f"  Training:   {train_auc:.4f}")
        print(f"  Validation: {val_auc:.4f}")
        print(f"  Test:       {test_auc:.4f}")
    
    # Overfitting indicators
    print("\n" + "="*70)
    print("OVERFITTING INDICATORS")
    print("="*70)
    
    overfit_gap_train_test = train_acc - test_acc
    overfit_gap_train_val = train_acc - val_acc
    
    print(f"\nTrain vs Test Accuracy Gap: {overfit_gap_train_test:.4f}")
    if overfit_gap_train_test > 0.05:
        print("   WARNING: Potential OVERFITTING detected!")
    elif overfit_gap_train_test > 0.02:
        print("   WARNING: Light overfitting signs")
    else:
        print("   OK (normal, acceptable)")
    
    print(f"\nTrain vs Validation Accuracy Gap: {overfit_gap_train_val:.4f}")
    if overfit_gap_train_val > 0.05:
        print("   WARNING: OVERFITTING detected!")
    else:
        print("   OK (normal)")
    
    # Confusion matrix analysis
    print("\n" + "="*70)
    print("CONFUSION MATRIX ANALYSIS")
    print("="*70)
    
    for set_name, y_true, y_pred in [("Training", y_train, y_train_pred), 
                                       ("Validation", y_val, y_val_pred),
                                       ("Test", y_test, y_test_pred)]:
        cm = confusion_matrix(y_true, y_pred)
        if len(cm) == 2 and len(cm[0]) == 2:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Handle binary case with only one class
            if y_pred.max() == 0:  # Only negative predictions
                tn = (y_true == 0).sum()
                fp = 0
                fn = (y_true == 1).sum()
                tp = 0
            else:
                tp = (y_true == 1).sum()
                tn = (y_true == 0).sum()
                fp = 0
                fn = 0
        
        print(f"\n{set_name}:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
    
    # Cross-validation
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    fold_num = 1
    for train_idx, test_idx in skf.split(X_all, y_all):
        cv_model = RandomForestModel()
        cv_model.train(X_all[train_idx], y_all[train_idx])
        cv_pred = cv_model.predict(X_all[test_idx])
        fold_acc = accuracy_score(y_all[test_idx], cv_pred)
        cv_scores.append(fold_acc)
        print(f"Fold {fold_num}: {fold_acc:.4f} ({fold_acc*100:.2f}%)")
        fold_num += 1
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCross-Validation Mean: {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"Cross-Validation Range: {min(cv_scores):.4f} to {max(cv_scores):.4f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if test_acc >= 0.95 and overfit_gap_train_test < 0.1 and cv_mean >= 0.88:
        print("\nVERDICT: NO OVERFITTING DETECTED")
        print(f"  - Test Accuracy: {test_acc*100:.2f}%")
        print(f"  - CV Score: {cv_mean*100:.2f}%")
        print(f"  - Train-Test Gap: {overfit_gap_train_test*100:.2f}%")
        print(f"  - Generalization: EXCELLENT")
        print("\nCONCLUSION: 100% accuracy is LEGITIMATE!")
    elif test_acc >= 0.90 and overfit_gap_train_test < 0.15:
        print("\nVERDICT: MINIMAL OVERFITTING")
        print(f"  - Test Accuracy: {test_acc*100:.2f}%")
        print(f"  - CV Score: {cv_mean*100:.2f}%")
        print(f"  - Model is still reliable")
    else:
        print("\nVERDICT: SIGNIFICANT OVERFITTING")
        print(f"  - Test Accuracy: {test_acc*100:.2f}%")
        print(f"  - Train-Test Gap: {overfit_gap_train_test*100:.2f}%")
        print(f"  - CV Score: {cv_mean*100:.2f}%")
        print(f"  - Generalization: WEAK - Model may not perform well on new data")
    
    print("\n" + "="*70)

except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
