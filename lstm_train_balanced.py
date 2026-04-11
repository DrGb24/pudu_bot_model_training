#!/usr/bin/env python3
"""
LSTM Training with Class Weights to Handle Imbalanced Data
Başarısızlıkları ağırlıklandarak eğitim
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG, DATA_CONFIG, MODEL_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from kpi_metrics import KPIMetrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Setup logging
LOG_DIR = Path('logs/lstm')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'training_balanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load snapshot data
logger.info("📊 Loading training data snapshot...")
df = pd.read_csv('data/lstm_database_snapshot.csv')
logger.info(f"✅ Data loaded: {df.shape}")
logger.info(f"\n📈 Class Distribution:")
logger.info(f"   Normal (0): {(df['failure']==0).sum()} ({(df['failure']==0).sum()/len(df)*100:.1f}%)")
logger.info(f"   Failure (1): {(df['failure']==1).sum()} ({(df['failure']==1).sum()/len(df)*100:.1f}%)")

# Prepare features
features = df.drop('failure', axis=1).values
target = df['failure'].values

# Scale features
scaler = StandardScaler()
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

# Split data (70/15/15)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logger.info(f"\n📊 Data Split:")
logger.info(f"   Train: {X_train.shape} | Failures: {y_train.sum()}")
logger.info(f"   Val:   {X_val.shape} | Failures: {y_val.sum()}")
logger.info(f"   Test:  {X_test.shape} | Failures: {y_test.sum()}")

# Calculate class weights
# weight = (total_samples) / (2 * num_class_samples)
n_samples = len(y_train)
n_failures = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

class_weight_failure = n_samples / (2 * n_failures) if n_failures > 0 else 1
class_weight_normal = n_samples / (2 * n_normal) if n_normal > 0 else 1

class_weights = {0: class_weight_normal, 1: class_weight_failure}

logger.info(f"\n⚖️  Class Weights:")
logger.info(f"   Normal (0): {class_weight_normal:.2f}")
logger.info(f"   Failure (1): {class_weight_failure:.2f}")

# Build LSTM model
logger.info(f"\n🔨 Building LSTM model...")
lstm_model = LSTMModel(input_shape=(sequence_length, 9))
lstm_model.build_model()

# Train with class weights
logger.info(f"\n⏳ Training LSTM with class weights...")
history = lstm_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=0,
    class_weight=class_weights  # 🔑 Pass class weights
)

# Evaluate
logger.info(f"\n📈 Evaluating model...")
results = lstm_model.evaluate(X_test, y_test)

# Save model
logger.info(f"\n💾 Saving improved model...")
model_path = Path('models/lstm/lstm_model_balanced.h5')
lstm_model.save_model(str(model_path))
logger.info(f"✅ Model saved: {model_path}")

# Save scaler
scaler_path = Path('models/lstm/lstm_scaler.pkl')
joblib.dump(scaler, str(scaler_path))

# Generate report
report = {
    'Metric': [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
        'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
        'Class Weight (Normal)', 'Class Weight (Failure)', 'Training Epochs'
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
        f"{class_weight_normal:.2f}",
        f"{class_weight_failure:.2f}",
        str(len(history['loss']))
    ]
}

report_df = pd.DataFrame(report)
report_path = Path('logs/lstm/lstm_report_balanced.csv')
report_df.to_csv(report_path, index=False)
logger.info(f"✅ Report saved: {report_path}")

logger.info(f"\n{'='*70}")
logger.info(f"✅ BALANCED LSTM TRAINING COMPLETED")
logger.info(f"{'='*70}")
