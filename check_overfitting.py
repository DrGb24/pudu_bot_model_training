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
print("OVERFITTING ANALİZİ - %100 Accuracy Doğrulaması")
print("="*70)

try:
    # Load data from database
    print("\n1️⃣ VERİ YÜKLENİYOR...")
    data_prep = DataPreparation()
    df = data_prep.load_from_database(
        db_config=DATABASE_CONFIG,
        query="""
        SELECT 
            (1 - is_success::int) as failure,
            check_result_count as error_count,
            EXTRACT(EPOCH FROM (now() - task_time))/3600 as operational_hours,
            RANDOM()*100 as temperature,
            RANDOM() as vibration,
            RANDOM()*150 as pressure,
            RANDOM()*100 as humidity,
            (RANDOM()*10000)::int as last_maintenance_days,
            (RANDOM()*120)::int as robot_age_months,
            RANDOM()*1000 as power_consumption
        FROM robot_logs_info
        WHERE check_result_count > 0
        LIMIT 2000
        """
    )
    print(f"✅ {len(df)} satır yüklendi")
    print(f"Target dağılımı:\n{df['failure'].value_counts()}\n")
    
    # Prepare data
    print("2️⃣ VERİ HAZIRLANIYYOR...")
    df.to_csv('data/temp_analysis_data.csv', index=False)
    
    X_train, X_val, X_test, y_train, y_val, y_test, features = data_prep.prepare_data(
        filepath='data/temp_analysis_data.csv',
        target_column=DATA_CONFIG['target_column'],
        categorical_cols=DATA_CONFIG['categorical_columns'],
        numerical_cols=DATA_CONFIG['numerical_columns'],
        validation_size=MODEL_CONFIG.get('validation_size', 0.15),
        return_validation=True
    )
    
    print(f"\n📊 Veri Bölümü:")
    print(f"  Training:   {X_train.shape[0]} örnek ({X_train.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"  Validation: {X_val.shape[0]} örnek ({X_val.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    print(f"  Test:       {X_test.shape[0]} örnek ({X_test.shape[0]/(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])*100:.1f}%)")
    
    # Train model
    print("\n3️⃣ MODEL EĞİTİLİYOR...")
    model = RandomForestModel()
    model.train(X_train, y_train)
    print("✅ Model eğitim tamamlandı")
    
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
    print("📊 OVERFITTING ANALİZİ SONUÇLARI")
    print("="*70)
    
    print("\n🎯 ACCURACY (Doğruluk):")
    print(f"  Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Validation: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"  Test:       {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    print("\n🎯 PRECISION (Kesinlik):")
    print(f"  Training:   {train_prec:.4f}")
    print(f"  Validation: {val_prec:.4f}")
    print(f"  Test:       {test_prec:.4f}")
    
    print("\n🎯 RECALL (Hatırlanma):")
    print(f"  Training:   {train_rec:.4f}")
    print(f"  Validation: {val_rec:.4f}")
    print(f"  Test:       {test_rec:.4f}")
    
    print("\n🎯 F1-SCORE:")
    print(f"  Training:   {train_f1:.4f}")
    print(f"  Validation: {val_f1:.4f}")
    print(f"  Test:       {test_f1:.4f}")
    
    if train_auc > 0:
        print("\n🎯 AUC-ROC (ROC Eğrisi Altındaki Alan):")
        print(f"  Training:   {train_auc:.4f}")
        print(f"  Validation: {val_auc:.4f}")
        print(f"  Test:       {test_auc:.4f}")
    
    # Overfitting indicators
    print("\n" + "="*70)
    print("⚠️  OVERFITTING GÖSTERGELERI")
    print("="*70)
    
    overfit_gap_train_test = train_acc - test_acc
    overfit_gap_train_val = train_acc - val_acc
    
    print(f"\n🔴 Train vs Test Accuracy Farkı: {overfit_gap_train_test:.4f}")
    if overfit_gap_train_test > 0.05:
        print("   ⚠️  UYARI: Potansiyel OVERFITTING Var!")
    elif overfit_gap_train_test > 0.02:
        print("   ⚠️  UYARI: Hafif OVERFITTING belirtileri")
    else:
        print("   ✅ Normal (acceptable)")
    
    print(f"\n🔴 Train vs Validation Accuracy Farkı: {overfit_gap_train_val:.4f}")
    if overfit_gap_train_val > 0.05:
        print("   ⚠️  UYARI: OVERFITTING Var!")
    else:
        print("   ✅ Normal")
    
    # Confusion matrix analysis
    print("\n" + "="*70)
    print("📋 CONFUSION MATRIX ANALİZİ")
    print("="*70)
    
    for set_name, y_true, y_pred in [("Training", y_train, y_train_pred), 
                                       ("Validation", y_val, y_val_pred),
                                       ("Test", y_test, y_test_pred)]:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"\n{set_name}:")
        print(f"  True Positives:  {tp}")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
    
    # Cross-validation
    print("\n" + "="*70)
    print("🔄 5-FOLD CROSS-VALIDATION")
    print("="*70)
    
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    
    cv_model = RandomForestModel()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
        cv_model.train(X_all[train_idx], y_all[train_idx])
        cv_pred = cv_model.predict(X_all[test_idx])
        fold_acc = accuracy_score(y_all[test_idx], cv_pred)
        cv_scores.append(fold_acc)
        print(f"Fold {fold}: {fold_acc:.4f} ({fold_acc*100:.2f}%)")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nCross-Validation Ortalaması: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Final verdict
    print("\n" + "="*70)
    print("🔍 SONUÇ VE DEĞERLENDİRME")
    print("="*70)
    
    if test_acc >= 0.95 and overfit_gap_train_test < 0.1 and cv_mean >= 0.85:
        print("\n✅ TEST BAŞARILI - Overfitting YOK")
        print(f"   • Test Accuracy: {test_acc*100:.2f}%")
        print(f"   • CV Score: {cv_mean*100:.2f}%")
        print(f"   • Generalization: GOOD")
    else:
        print("\n⚠️  OVERFITTING İHTİMALİ YÜKSEK")
        print(f"   • Test Accuracy: {test_acc*100:.2f}%")
        print(f"   • Train-Test Gap: {overfit_gap_train_test*100:.2f}%")
        print(f"   • CV Score: {cv_mean*100:.2f}%")
        print(f"   • Generalization: WEAK")
    
    print("\n" + "="*70)

except Exception as e:
    print(f"❌ Hata: {str(e)}")
    import traceback
    traceback.print_exc()
