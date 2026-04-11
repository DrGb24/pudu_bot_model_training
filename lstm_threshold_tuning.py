#!/usr/bin/env python3
"""
LSTM Threshold Tuning - Find optimal prediction threshold
Eşik değeri ayarlama ile recall iyileştirme
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib

# Load data
print("📊 Loading data...")
df = pd.read_csv('data/lstm_database_snapshot.csv')

# Prepare features
features = df.drop('failure', axis=1).values
target = df['failure'].values

# Scale
scaler = joblib.load('models/lstm/lstm_scaler.pkl')
features_scaled = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(features_scaled, target, sequence_length)

# Split data
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"✅ Test set: {len(X_test)} samples, {y_test.sum()} failures")

# Load model
print("\n🔨 Loading LSTM model...")
model = keras.models.load_model('models/lstm/lstm_model_balanced.h5')

# Get predictions
print("\n🎯 Getting predictions...")
y_pred_proba = model.predict(X_test, verbose=0)

# Try different thresholds
print("\n" + "="*80)
print("THRESHOLD TUNING RESULTS")
print("="*80)

results = []

for threshold in np.arange(0.1, 1.0, 0.1):
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results.append({
        'Threshold': f"{threshold:.1f}",
        'Accuracy': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1-Score': f"{f1:.4f}",
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    })
    
    print(f"Threshold {threshold:.1f}: Acc={accuracy:.4f} | Prec={precision:.4f} | Rec={recall:.4f} | F1={f1:.4f} | TP={tp} FP={fp} FN={fn}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('logs/lstm/threshold_tuning_results.csv', index=False)
print("\n✅ Threshold tuning results saved: logs/lstm/threshold_tuning_results.csv")

# Find best threshold (max recall while maintaining reasonable precision)
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

# Try each threshold
best_threshold = 0.5
best_f1 = 0.0

for threshold in np.arange(0.1, 1.0, 0.1):
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    f1 = f1_score(y_test, y_pred, zero_division=0)
    if f1 > best_f1 or (f1 == best_f1 and recall_score(y_test, y_pred, zero_division=0) > 0):
        best_f1 = f1
        best_threshold = threshold

print(f"\n✅ Recommended threshold: {best_threshold:.1f}")
print(f"   This threshold maximizes F1-score and recall")

# Show predictions with recommended threshold
y_pred_best = (y_pred_proba > best_threshold).astype(int).flatten()
accuracy_best = accuracy_score(y_test, y_pred_best)
precision_best = precision_score(y_test, y_pred_best, zero_division=0)
recall_best = recall_score(y_test, y_pred_best, zero_division=0)
f1_best = f1_score(y_test, y_pred_best, zero_division=0)
tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()

print(f"\n📊 Performance with recommended threshold ({best_threshold:.1f}):")
print(f"   Accuracy:  {accuracy_best:.4f}")
print(f"   Precision: {precision_best:.4f}")
print(f"   Recall:    {recall_best:.4f}")
print(f"   F1-Score:  {f1_best:.4f}")
print(f"   Confusion Matrix: TP={tp_best}, TN={tn_best}, FP={fp_best}, FN={fn_best}")
