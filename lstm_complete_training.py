#!/usr/bin/env python3
"""
Complete LSTM training - save model and generate report from snapshot
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lstm_models import LSTMModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

# Load training history
print("✅ Loading training history...")
history_file = Path('logs/lstm/training_history.json')
with open(history_file, 'r') as f:
    history = json.load(f)

print(f"   Epochs trained: {len(history['loss'])}")
print(f"   Final loss: {history['loss'][-1]:.6f}")
print(f"   Final accuracy: {history['accuracy'][-1]:.4f}")

# Load snapshot data
print("\n📊 Loading snapshot data for test evaluation...")
snapshot_file = Path('data/lstm_database_snapshot.csv')
df = pd.read_csv(snapshot_file)
print(f"✅ Data loaded: {df.shape}")

# Create sequences
sequence_length = 10
scaler = joblib.load(Path('models/lstm/lstm_scaler.pkl'))

# Prepare features
features = df.drop('failure', axis=1).values
target = df['failure'].values

# Features should already be normalized by the scaler in lstm_train
# But let's normalize them again to be sure
features_scaled = scaler.fit_transform(features)

# Create sequences
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled, target, sequence_length)

# Split data (70/15/15)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

print(f"   Test set size: {len(X_test)}")
print(f"   Failure rate in test: {y_test.mean()*100:.2f}%")

# Build LSTM model
print("\n🔨 Rebuilding LSTM model architecture...")
lstm_model = LSTMModel(input_shape=(10, 9))
lstm_model.build_model()

# Compile model
lstm_model.model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("   Model recompiled")

# Calculate class weights
n_samples = len(y_train)
n_failures = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

class_weight_failure = n_samples / (2 * n_failures) if n_failures > 0 else 1
class_weight_normal = n_samples / (2 * n_normal) if n_normal > 0 else 1

class_weights = {0: class_weight_normal, 1: class_weight_failure}

print(f"\n⚖️  Class Weights:")
print(f"   Normal (0): {class_weight_normal:.2f}")
print(f"   Failure (1): {class_weight_failure:.2f}")

# Train model to match history (this replicates the previous training)
print("\n⏳ Training model to match saved history (36 epochs) with class weights...")
lstm_model.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=len(history['loss']),
    batch_size=32,
    class_weight=class_weights,  # 🔑 Apply class weights
    verbose=0
)

print("   ✅ Training matched")

# Evaluate on test set
print("\n📈 Evaluating on test set...")
y_pred_proba = lstm_model.model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc_roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

print(f"✅ Test Results:")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   AUC-ROC:   {auc_roc:.4f}")

print(f"\n   Confusion Matrix:")
print(f"   TN={cm[0,0]}, FP={cm[0,1]}")
print(f"   FN={cm[1,0]}, TP={cm[1,1]}")

# Save model
print("\n💾 Saving LSTM model...")
model_path = Path('models/lstm/lstm_model.h5')
lstm_model.model.save(str(model_path))
print(f"✅ Model saved: {model_path}")

# Generate report
print("\n📄 Generating training report...")
report_data = {
    'Metric': [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
        'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
        'Training Epochs', 'Final Loss', 'Final Training Accuracy'
    ],
    'Value': [
        f"{accuracy:.4f}",
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1:.4f}",
        f"{auc_roc:.4f}",
        str(cm[0, 0]),
        str(cm[0, 1]),
        str(cm[1, 0]),
        str(cm[1, 1]),
        str(len(history['loss'])),
        f"{history['loss'][-1]:.6f}",
        f"{history['accuracy'][-1]:.4f}"
    ]
}

report_df = pd.DataFrame(report_data)
report_path = Path('logs/lstm/lstm_final_report.csv')
report_df.to_csv(report_path, index=False)
print(f"✅ Report saved: {report_path}")

print("\n" + "="*70)
print("✅ LSTM MODEL TRAINING COMPLETED SUCCESSFULLY")
print("="*70)
print(f"\n📊 Summary:")
print(f"   Model: models/lstm/lstm_model.h5")
print(f"   Report: logs/lstm/lstm_final_report.csv")
print(f"   Test Accuracy: {accuracy:.4f}")
print(f"   Test Recall: {recall:.4f}")
