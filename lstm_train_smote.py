#!/usr/bin/env python3
"""
LSTM Training with SMOTE - Synthetic Minority Over-sampling
Başarısızlıkları yapay olarak çoğalt
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import logging

# Setup logging
LOG_DIR = Path('logs/lstm')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'training_smote_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load data
logger.info("📊 Loading data...")
df = pd.read_csv('data/lstm_database_snapshot.csv')
logger.info(f"✅ Data loaded: {df.shape}")
logger.info(f"   Failures: {(df['failure']==1).sum()} ({(df['failure']==1).sum()/len(df)*100:.1f}%)")

# Prepare features
features = df.drop('failure', axis=1).values
target = df['failure'].values

# Standardize
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply SMOTE BEFORE creating sequences
logger.info("\n🔄 Applying SMOTE (over-sampling failures)...")
smote = SMOTE(random_state=42, k_neighbors=3)
features_resampled, target_resampled = smote.fit_resample(features_scaled, target)

logger.info(f"✅ After SMOTE:")
logger.info(f"   Total samples: {len(target_resampled)}")
logger.info(f"   Normal: {(target_resampled==0).sum()}")
logger.info(f"   Failures: {(target_resampled==1).sum()}")

# Create sequences from resampled data
def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(features_resampled, target_resampled, sequence_length)

# Split data (70/15/15)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logger.info(f"\n📊 Data Split (after resampling):")
logger.info(f"   Train: {X_train.shape} | Failures: {y_train.sum()}")
logger.info(f"   Val:   {X_val.shape} | Failures: {y_val.sum()}")
logger.info(f"   Test:  {X_test.shape} | Failures: {y_test.sum()}")

# Build LSTM
logger.info(f"\n🔨 Building LSTM model...")
lstm_model = LSTMModel(input_shape=(sequence_length, 9))
lstm_model.build_model()

# Train WITHOUT class weights (data is already balanced)
logger.info(f"\n⏳ Training LSTM (data is balanced, no class weights needed)...")
history = lstm_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=0
)

logger.info(f"✅ Training completed: {len(history['loss'])} epochs")

# Evaluate on ORIGINAL test set (not resampled)
logger.info(f"\n📊 Creating test set from original data (for fair evaluation)...")

# Create test sequences from original scaled data
X_orig, y_orig = create_sequences(features_scaled, target, sequence_length)
_, _, X_test_orig, y_test_orig = (
    X_orig[:train_size], y_orig[:train_size],
    X_orig[train_size+val_size:], y_orig[train_size+val_size:]
)

logger.info(f"   Test set (original): {X_test_orig.shape}")
logger.info(f"   Failures in test:  {y_test_orig.sum()}")

# Evaluate
logger.info(f"\n📈 Evaluating model on original test set...")
results = lstm_model.evaluate(X_test_orig, y_test_orig)

# Save model
logger.info(f"\n💾 Saving SMOTE-trained model...")
model_path = Path('models/lstm/lstm_model_smote.h5')
lstm_model.save_model(str(model_path))

scaler_path = Path('models/lstm/lstm_scaler.pkl')
joblib.dump(scaler, str(scaler_path))

# Generate report
report = {
    'Metric': [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
        'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
        'Training Method', 'Training Epochs', 'Test Failures Detected'
    ],
    'Value': [
        f"{results['accuracy']:.4f}",
        f"{results['precision']:.4f}",
        f"{results['recall']:.4f}",
        f"{results['f1_score']:.4f}",
        f"{results['auc_roc']:.4f}",
        str(results['true_negatives']),
        str(results['false_positives']),
        str(results['false_negatives']),
        str(results['true_positives']),
        "SMOTE (Synthetic Over-sampling)",
        str(len(history['loss'])),
        str(results['true_positives'] + results['false_negatives'])  # Total failures
    ]
}

report_df = pd.DataFrame(report)
report_path = Path('logs/lstm/lstm_report_smote.csv')
report_df.to_csv(report_path, index=False)
logger.info(f"✅ Report saved: {report_path}")

logger.info(f"\n{'='*70}")
logger.info(f"✅ SMOTE-TRAINED LSTM COMPLETED")
logger.info(f"{'='*70}")
