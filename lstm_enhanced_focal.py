#!/usr/bin/env python3
"""
LSTM ENHANCED TRAINING - Sentetik Veri + Focal Loss
Real + Synthetic 15K dataset ile Focal Loss kullanan improved LSTM
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall

# Setup logging
LOG_DIR = Path('logs/lstm')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'training_enhanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("LSTM ENHANCEDLEMİŞ TRAINING - SENTETIK VERİ + FOCAL LOSS")
print("="*80)

# CSV'den veri yükle
logger.info("📊 Deniz veri yükleniyor (Real + Synthetic)...")
csv_path = Path('data/lstm_combined_15k.csv')
if not csv_path.exists():
    logger.error(f"❌ Dosya bulunamadı: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
logger.info(f"✅ Veri yüklendi: {df.shape}")
logger.info(f"   Başarısızlıklar: {(df['failure'] == 1).sum()} ({df['failure'].mean()*100:.2f}%)")
logger.info(f"   Başarılılar: {(df['failure'] == 0).sum()}")

# Features ve target
features = df[['error_count', 'task_hour', 'day_of_month', 'day_of_week',
               'robot_id_length', 'software_version_length', 'product_code_type',
               'error_severity', 'hourly_error_rate']].values
target = df['failure'].values

# Scale
logger.info("📈 Veriler normalize ediliyor...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Save scaler
scaler_path = Path('models/lstm/lstm_scaler_enhanced.pkl')
scaler_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(scaler, scaler_path)
logger.info(f"💾 Scaler kaydedildi: {scaler_path}")

# Create sequences
sequence_length = 10
sequences = []
labels = []

for i in range(len(features_scaled) - sequence_length + 1):
    seq = features_scaled[i:i+sequence_length]
    label = target[i + sequence_length - 1]
    sequences.append(seq)
    labels.append(label)

X = np.array(sequences, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

logger.info(f"✅ Sequence'ler oluşturuldu: {X.shape}")
logger.info(f"   Başarısızlıklar: {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")

# Train/Val/Test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.175, random_state=42, stratify=y_temp
)

logger.info(f"\n📊 Veri Split:")
logger.info(f"   Train: {X_train.shape} | Başarısızlıklar: {(y_train == 1).sum()}")
logger.info(f"   Val:   {X_val.shape} | Başarısızlıklar: {(y_val == 1).sum()}")
logger.info(f"   Test:  {X_test.shape} | Başarısızlıklar: {(y_test == 1).sum()}")

# SMOTE on training data
logger.info("\n🔄 SMOTE over-sampling (training set)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train.reshape(X_train.shape[0], -1),
    y_train
)
# Reshape back to sequences
X_train_resampled = X_train_resampled.reshape(-1, sequence_length, X.shape[2])

logger.info(f"✅ SMOTE tamamlandı:")
logger.info(f"   Öncesi: {X_train.shape} | {(y_train == 1).sum()} failures")
logger.info(f"   Sonrası: {X_train_resampled.shape} | {(y_train_resampled == 1).sum()} failures")

# Build enhanced LSTM with Focal Loss
logger.info("\n🔨 Enhanced LSTM Modeli Oluşturuluyor (Focal Loss)...")

def focal_loss(gamma=2., alpha=0.25):
    """Focal Loss for addressing class imbalance"""
    def focal_crossentropy(y_true, y_pred):
        # Clip predictions
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate cross entropy
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal weight
        p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        # Apply focal weight and alpha
        focal_ce = alpha * focal_weight * ce
        
        return tf.reduce_mean(focal_ce)
    
    return focal_crossentropy

# Build model
model = keras.Sequential([
    layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2),
                        input_shape=(sequence_length, X.shape[2])),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
    layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.2)),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

logger.info(f"✅ LSTM Model built. Total params: {model.count_params():,}")

# Callbacks
logger.info("\n⏳ LSTM Training (100 epochs, Focal Loss, SMOTE)...")

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train
history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight={0: 0.5, 1: 5.0},  # Daha az aggressive weight
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

logger.info(f"✅ Training tamamlandı: {len(history.history['loss'])} epochs")

# Evaluate
logger.info("\n📈 Model Değerlendiriliyor...")
test_loss, test_acc, test_prec, test_rec, test_auc = model.evaluate(
    X_test, y_test, verbose=0
)

# Predictions
y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate metrics
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
auc_roc = roc_auc_score(y_test, y_pred_prob)

logger.info("\n" + "="*50)
logger.info("LSTM ENHANCED MODEL EVALUATION")
logger.info("="*50)
logger.info(f"  Accuracy:  {test_acc:.4f}")
logger.info(f"  Precision: {precision:.4f}")
logger.info(f"  Recall:    {recall:.4f}")
logger.info(f"  F1-Score:  {f1:.4f}")
logger.info(f"  AUC-ROC:   {auc_roc:.4f}")
logger.info("="*50)

# Save model
logger.info("\n💾 Model Kaydediliyor...")
model_path = Path('models/lstm/lstm_enhanced_focal.h5')
model_path.parent.mkdir(parents=True, exist_ok=True)

# Weights only kaydına çevir
model.save_weights(str(model_path.with_suffix('.weights.h5')))
logger.info(f"✅ Model weights saved: {model_path.with_suffix('.weights.h5')}")

# Architecture'ı JSON olarak kaydet
import json
with open(model_path.with_suffix('.json'), 'w') as f:
    json.dump(model.to_json(), f)
logger.info(f"✅ Model architecture saved: {model_path.with_suffix('.json')}")

# Legacy compatibility için HDF5 de kaydet (compile=False)
try:
    model.save(model_path, save_format='h5')
    logger.info(f"✅ Model saved (legacy HDF5): {model_path}")
except Exception as e:
    logger.warning(f"⚠️ Legacy HDF5 save hatası (normal): {e}")

# Save training history
history_path = Path('logs/lstm/training_history_enhanced.json')
with open(history_path, 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

# Generate report
report = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
               'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
               'Training Epochs', 'Training Samples', 'Test Failures'],
    'Value': [test_acc, precision, recall, f1, auc_roc,
              tn, fp, fn, tp,
              len(history.history['loss']), X_train_resampled.shape[0], (y_test == 1).sum()]
}

report_df = pd.DataFrame(report)
report_path = Path('logs/lstm/lstm_report_enhanced.csv')
report_df.to_csv(report_path, index=False)
logger.info(f"✅ Rapor Kaydedildi: {report_path}")

logger.info("\n" + "="*80)
logger.info("✅ ENHANCED LSTM TRAİNİNG TAMAMLANDI")
logger.info("="*80)

logger.info(f"\n📊 Özet:")
logger.info(f"   Model Dosyası: {model_path}")
logger.info(f"   Rapor Dosyası: {report_path}")
logger.info(f"   Combined Veri: 14,993 örnek (real + synthetic)")
logger.info(f"   SMOTE After: {X_train_resampled.shape[0]:,} training samples")
logger.info(f"   Test Başarısızlıkları: {(y_test == 1).sum()}")
logger.info(f"   Recall: {recall:.4f}")
logger.info(f"   Precision: {precision:.4f}")
