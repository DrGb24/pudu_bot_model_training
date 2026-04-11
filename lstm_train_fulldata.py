#!/usr/bin/env python3
"""
LSTM YENİDEN EĞİT - TÜM DATABASE VERİLERİ İLE (204 başarısızlık)
Tüm 3,893 örneğini kullan
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
LOG_DIR = Path('logs/lstm')
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'training_fulldata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("LSTM YENİDEN EĞİTİ - TÜM DATABASE VERİLERİ İLE")
print("="*80)

# Database'den TÜM verileri al (LIMIT KALDIRIYORUZ!)
logger.info("📊 Database'den TÜM verileri yükleniyor...")
data_prep = DataPreparation(DATABASE_CONFIG)

# TÜMÜNÜ AL - LIMIT KALDIRIYORUZ!
sql_query = """
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
           WHEN product_code LIKE '%Bellabot%' THEN 3
           WHEN product_code LIKE '%CC%' THEN 4
           ELSE 5 END as product_code_type,
      check_result_count as error_severity,
      COALESCE(0, 0) as hourly_error_rate
    FROM robot_logs_info
    ORDER BY RANDOM()
"""

df = data_prep.load_from_database(DATABASE_CONFIG, query=sql_query)
logger.info(f"✅ Toplam veri yüklendi: {df.shape}")
logger.info(f"   Başarısızlıklar: {df['failure'].sum()} ({df['failure'].sum()/len(df)*100:.2f}%)")
logger.info(f"   Başarılılar: {len(df) - df['failure'].sum()}")

# Prepare features
features = df.drop('failure', axis=1).values
target = df['failure'].values

# Scale
logger.info("📈 Veriler normalize ediliyor...")
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

logger.info(f"✅ Sequence'ler oluşturuldu: {X.shape}")
logger.info(f"   Başarısızlık sayısı: {y.sum()}")

# Split data (70/15/15)
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

logger.info(f"\n📊 Veri Split:")
logger.info(f"   Train: {X_train.shape} | Başarısızlıklar: {y_train.sum()}")
logger.info(f"   Val:   {X_val.shape} | Başarısızlıklar: {y_val.sum()}")
logger.info(f"   Test:  {X_test.shape} | Başarısızlıklar: {y_test.sum()}")

# Calculate class weights
n_samples = len(y_train)
n_failures = (y_train == 1).sum()
n_normal = (y_train == 0).sum()

class_weight_failure = n_samples / (2 * n_failures) if n_failures > 0 else 1
class_weight_normal = n_samples / (2 * n_normal) if n_normal > 0 else 1

class_weights = {0: class_weight_normal, 1: class_weight_failure}

logger.info(f"\n⚖️  Sınıf Ağırlıkları:")
logger.info(f"   Normal: {class_weight_normal:.2f}")
logger.info(f"   Başarısızlık: {class_weight_failure:.2f}")

# Build LSTM
logger.info(f"\n🔨 LSTM Modeli Oluşturuluyor...")
lstm_model = LSTMModel(input_shape=(sequence_length, 9))
lstm_model.build_model()

# Train
logger.info(f"\n⏳ LSTM Eğitimi Başlıyor (100 epochs, class weights ile)...")
history = lstm_model.train(
    X_train, y_train,
    X_val, y_val,
    epochs=100,
    batch_size=32,
    verbose=0,
    class_weight=class_weights
)

logger.info(f"✅ Eğitim Tamamlandı: {len(history['loss'])} epochs")

# Evaluate
logger.info(f"\n📈 Model Değerlendiriliyor...")
results = lstm_model.evaluate(X_test, y_test)

# Save
logger.info(f"\n💾 Model Kaydediliyor...")
model_path = Path('models/lstm/lstm_model_fulldata.h5')
lstm_model.save_model(str(model_path))

scaler_path = Path('models/lstm/lstm_scaler.pkl')
joblib.dump(scaler, str(scaler_path))

# Report
report = {
    'Metric': [
        'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
        'True Negatives', 'False Positives', 'False Negatives', 'True Positives',
        'Training Epochs', 'Training Samples', 'Test Failures'
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
        str(len(history['loss'])),
        str(len(df)),
        str(y_test.sum())
    ]
}

report_df = pd.DataFrame(report)
report_path = Path('logs/lstm/lstm_report_fulldata.csv')
report_df.to_csv(report_path, index=False)
logger.info(f"✅ Rapor Kaydedildi: {report_path}")

print("\n" + "="*80)
print("✅ TÜM VERİ İLE LSTM EĞİTİMİ TAMAMLANDI")
print("="*80)
print(f"\n📊 ÖzetliBilgiler:")
print(f"   Model Dosyası: {model_path}")
print(f"   Rapor Dosyası: {report_path}")
print(f"   Eğitim Örnekleri: {len(df):,}")
print(f"   Test Başarısızlıkları: {y_test.sum()}")
print(f"   Recall: {results['recall']:.4f}")
print(f"   Precision: {results['precision']:.4f}")
