#!/usr/bin/env python3
"""
LSTM Model Usage Examples
Demonstrating all LSTM model capabilities with real and synthetic data
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import DATABASE_CONFIG
from data_preparation import DataPreparation
from lstm_models import LSTMModel, LSTMInference

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ÖRNEK 1: LSTM Model Mimarisi Gösterimi
# ============================================================================

def example_1_model_architecture():
    """Show LSTM architecture and parameters"""
    
    print("\n" + "="*70)
    print("ÖRNEK 1: LSTM Model Mimarisi")
    print("="*70 + "\n")
    
    # Define input shape (sequence_length=10, features=9)
    input_shape = (10, 9)
    
    # Create model
    lstm = LSTMModel(
        input_shape=input_shape,
        lstm_units=128,
        dropout_rate=0.2,
        dense_units=64,
        learning_rate=0.001
    )
    
    # Build model
    model = lstm.build_model()
    
    # Show summary
    print("\n📊 Model Özeti:")
    print(f"   Input Shape: {input_shape}")
    print(f"   Total Parameters: {model.count_params():,}")
    
    # Show config
    config = lstm.get_config()
    print("\n⚙️ Konfigürasyon:")
    for key, value in config.items():
        print(f"   {key}: {value}")


# ============================================================================
# ÖRNEK 2: Rastgele Veri ile LSTM Eğitimi
# ============================================================================

def example_2_lstm_training_synthetic():
    """Train LSTM on synthetic time-series data"""
    
    print("\n" + "="*70)
    print("ÖRNEK 2: Rastgele Veriler ile LSTM Eğitimi")
    print("="*70 + "\n")
    
    # Generate synthetic sequences
    print("📊 Rastgele veri seti oluşturuluyor...")
    sequence_length = 10
    num_features = 9
    num_samples = 500
    
    X_synthetic = np.random.randn(num_samples, sequence_length, num_features)
    y_synthetic = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    
    print(f"   Train set: {X_synthetic.shape}")
    print(f"   Target: {y_synthetic.shape}")
    
    # Split data
    split_idx = int(0.8 * len(X_synthetic))
    X_train = X_synthetic[:split_idx]
    y_train = y_synthetic[:split_idx]
    X_test = X_synthetic[split_idx:]
    y_test = y_synthetic[split_idx:]
    
    print(f"\n📈 Eğitim başlıyor (20 epoch)...")
    
    # Create and train model
    lstm = LSTMModel(
        input_shape=(sequence_length, num_features),
        lstm_units=64,
        dropout_rate=0.2,
        dense_units=32,
        learning_rate=0.001
    )
    
    model = lstm.build_model()
    history = lstm.train(
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    print(f"\n📊 Sonuçlar:")
    results = lstm.evaluate(X_test, y_test)
    print(f"   Accuracy:  {results['accuracy']:.4f}")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")


# ============================================================================
# ÖRNEK 3: Gerçek Database Verisi ile LSTM Eğitimi (Simüle)
# ============================================================================

def example_3_lstm_with_database_simulation():
    """Simulate LSTM training with database data (without actual DB)"""
    
    print("\n" + "="*70)
    print("ÖRNEK 3: Database Simülasyonu ile LSTM Eğitimi")
    print("="*70 + "\n")
    
    print("💾 Database simülasyonu yapılıyor...")
    
    # Simulate database data (2000 samples like the real data)
    np.random.seed(42)
    n_samples = 2000
    
    # Create realistic features
    data = {
        'error_count': np.random.poisson(2, n_samples),
        'task_hour': np.random.randint(0, 24, n_samples),
        'task_day_of_month': np.random.randint(1, 31, n_samples),
        'task_day_of_week': np.random.randint(0, 7, n_samples),
        'robot_id_length': np.random.randint(5, 10, n_samples),
        'software_version_length': np.random.randint(3, 8, n_samples),
        'product_code_type': np.random.randint(1, 5, n_samples),
        'error_severity': np.random.poisson(1, n_samples),
        'hourly_error_rate': np.random.exponential(0.5, n_samples),
        'failure': np.random.choice([0, 1], n_samples, p=[0.976, 0.024])  # 2.4% failures
    }
    
    df = pd.DataFrame(data)
    print(f"   Generated {len(df)} samples")
    print(f"   Failure rate: {df['failure'].mean()*100:.1f}%")
    
    # Create sequences
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = [c for c in df.columns if c != 'failure']
    X_data = df[feature_cols].values
    y_data = df['failure'].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)
    
    # Create sequences
    sequence_length = 10
    X_seq = []
    y_seq = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y_data[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"   Sequences: {X_seq.shape}")
    
    # Split
    train_idx = int(0.7 * len(X_seq))
    val_idx = int(0.85 * len(X_seq))
    
    X_train = X_seq[:train_idx]
    y_train = y_seq[:train_idx]
    X_val = X_seq[train_idx:val_idx]
    y_val = y_seq[train_idx:val_idx]
    X_test = X_seq[val_idx:]
    y_test = y_seq[val_idx:]
    
    print(f"\n📊 Veri bölünmesi:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Val:   {X_val.shape[0]} samples")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    # Train model
    print(f"\n🚀 Model eğitimi başlıyor (30 epoch)...")
    
    lstm = LSTMModel(
        input_shape=(sequence_length, len(feature_cols)),
        lstm_units=128,
        dropout_rate=0.2,
        dense_units=64,
        learning_rate=0.001
    )
    
    lstm.build_model()
    history = lstm.train(
        X_train, y_train,
        X_val, y_val,
        epochs=30,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate
    print(f"\n📊 Test Seti Sonuçları:")
    results = lstm.evaluate(X_test, y_test)
    print(f"   Accuracy:  {results['accuracy']:.4f} ✅")
    print(f"   Precision: {results['precision']:.4f}")
    print(f"   Recall:    {results['recall']:.4f}")
    print(f"   F1-Score:  {results['f1_score']:.4f}")
    print(f"   AUC-ROC:   {results['auc_roc']:.4f}")


# ============================================================================
# ÖRNEK 4: Tekli Tahmin (Single Sample Prediction)
# ============================================================================

def example_4_single_prediction():
    """Make prediction for single robot sequence"""
    
    print("\n" + "="*70)
    print("ÖRNEK 4: Tekli Robot Tahmini")
    print("="*70 + "\n")
    
    # Create dummy model and data
    np.random.seed(42)
    sequence_length = 10
    num_features = 9
    
    lstm = LSTMModel(
        input_shape=(sequence_length, num_features),
        lstm_units=64,
        dropout_rate=0.2
    )
    lstm.build_model()
    
    # Create sample sequence
    X_sample = np.random.randn(1, sequence_length, num_features)
    
    print("🤖 Robot tahmini yapılıyor...")
    print(f"   Sequence şekli: {X_sample.shape}")
    
    # Predict
    prediction = lstm.predict(X_sample)
    
    print(f"\n📊 Tahmin Sonucu:")
    print(f"   Başarısızlık Olasılığı: {prediction[0][0]:.4f}")
    print(f"   Risk Seviyesi: ", end="")
    
    if prediction[0][0] < 0.3:
        print("🟢 DÜŞÜK (< 30%)")
    elif prediction[0][0] < 0.7:
        print("🟡 ORTA (30-70%)")
    else:
        print("🔴 YÜKSEK (> 70%)")


# ============================================================================
# ÖRNEK 5: Toplu Tahmin (Batch Prediction)
# ============================================================================

def example_5_batch_prediction():
    """Make batch predictions for multiple sequences"""
    
    print("\n" + "="*70)
    print("ÖRNEK 5: Toplu Tahmin (Batch)")
    print("="*70 + "\n")
    
    np.random.seed(42)
    sequence_length = 10
    num_features = 9
    batch_size = 10
    
    # Create model and data
    lstm = LSTMModel(
        input_shape=(sequence_length, num_features),
        lstm_units=64,
        dropout_rate=0.2
    )
    lstm.build_model()
    
    # Generate batch
    X_batch = np.random.randn(batch_size, sequence_length, num_features)
    
    print(f"🤖 {batch_size} robot tahmini yapılıyor...")
    predictions = lstm.predict(X_batch)
    
    print(f"\n📊 Toplu Tahmin Sonuçları:\n")
    print("Robot | Olasılık | Risk Seviyesi")
    print("─" * 38)
    
    for i, pred in enumerate(predictions):
        prob = pred[0]
        if prob < 0.3:
            risk = "🟢 DÜŞÜK"
        elif prob < 0.7:
            risk = "🟡 ORTA"
        else:
            risk = "🔴 YÜKSEK"
        
        print(f"{i+1:>5} | {prob:>7.2%} | {risk}")
    
    # Statistics
    probs = predictions.flatten()
    print("\n" + "─" * 38)
    print(f"Ortalama:     {probs.mean():.4f}")
    print(f"Std Dev:      {probs.std():.4f}")
    print(f"Min:          {probs.min():.4f}")
    print(f"Max:          {probs.max():.4f}")
    print(f"Yüksek Risk:  {(probs > 0.7).sum()} robot")


# ============================================================================
# ÖRNEK 6: Özellik Önem Derecesi (Feature Importance)
# ============================================================================

def example_6_feature_importance():
    """Show approximate feature importance from LSTM weights"""
    
    print("\n" + "="*70)
    print("ÖRNEK 6: Özellik Önem Derecesi")
    print("="*70 + "\n")
    
    np.random.seed(42)
    sequence_length = 10
    num_features = 9
    
    lstm = LSTMModel(
        input_shape=(sequence_length, num_features),
        lstm_units=64,
        dropout_rate=0.2
    )
    lstm.build_model()
    
    feature_names = [
        'error_count',
        'task_hour',
        'task_day_of_month',
        'task_day_of_week',
        'robot_id_length',
        'software_version_length',
        'product_code_type',
        'error_severity',
        'hourly_error_rate'
    ]
    
    print("📊 Özellik önem dereceleri hesaplanıyor...")
    importance_dict = lstm.get_feature_importance(feature_names)
    
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 En Önemli Özellikler:\n")
    print("Sıra | Özellik                    | Önem    | Grafik")
    print("─" * 60)
    
    for rank, (feature, importance) in enumerate(sorted_features, 1):
        bar_length = int(importance * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"{rank:>3} | {feature:<26} | {importance:>6.1%} | {bar}")


# ============================================================================
# ÖRNEK 7: Model Karşılaştırması (RF vs LSTM)
# ============================================================================

def example_7_model_comparison():
    """Compare Random Forest and LSTM characteristics"""
    
    print("\n" + "="*70)
    print("ÖRNEK 7: Random Forest vs LSTM Karşılaştırması")
    print("="*70 + "\n")
    
    comparison = {
        'Özellik': [
            'Model Türü',
            'Veri Türü',
            'Doğruluk',
            'Hatırlanma',
            'Eğitim Süresi',
            'Çıkarım Hızı',
            'İnterpretiblite',
            'GPU İhtiyacı',
            'Zaman Serisi',
            'Hiperparametre'
        ],
        'Random Forest': [
            'Ensemble Ağaçları',
            'Vektör Özellikleri',
            '99.33%',
            '71.43%',
            '~2 saniye',
            '~5ms',
            '⭐⭐⭐ Yüksek',
            '❌ Hayır',
            '⭐ Zayıf',
            '⭐ Basit'
        ],
        'LSTM': [
            'Tekrarlayan Ağ',
            'Zaman Serileri (Seq)',
            'TBD (~90%)',
            'TBD (target 85%)',
            '~5-10 dakika',
            '~50ms',
            '⭐ Düşük',
            '✅ Evet (önerilir)',
            '⭐⭐⭐ Mükemmel',
            '⭐⭐⭐ Karmaşık'
        ]
    }
    
    df_compare = pd.DataFrame(comparison)
    
    print(df_compare.to_string(index=False))
    
    print("\n" + "─" * 80)
    print("\n💡 Ne zaman hangi modeli kullanmalı?")
    print("\n🌲 Random Forest kullan:")
    print("   • Kategorik veya karışık özellikler varsa")
    print("   • Hızlı eğitim gerekiyorsa")
    print("   • Model açıklanabilirliği önemliyse")
    print("   • Zaman serisi deseni yoksa")
    
    print("\n🧠 LSTM kullan:")
    print("   • Zaman serileriyle çalışmak gerekiyorsa")
    print("   • Uzun-dönem bağımlılıkları yakalamak istiyorsan")
    print("   • GPU vardıysa")
    print("   • Derin öğrenme deneyimi varsa")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples"""
    
    print("\n" + "="*70)
    print("🚀 LSTM MODEL ÖRNEK KÜTÜPHANESİ")
    print("="*70)
    
    try:
        # Run examples
        example_1_model_architecture()
        example_2_lstm_training_synthetic()
        example_3_lstm_with_database_simulation()
        example_4_single_prediction()
        example_5_batch_prediction()
        example_6_feature_importance()
        example_7_model_comparison()
        
        print("\n" + "="*70)
        print("✅ Tüm örnekler başarıyla tamamlandı!")
        print("="*70)
        print("\n📚 Sonraki Adımlar:")
        print("   1. Gerçek database verileriyle eğit: python lstm_train.py")
        print("   2. Tahminleme yap: python lstm_inference.py")
        print("   3. RF ve LSTM karşılaştır")
        print("   4. En iyi modeli production'a deploy et")
        
        return 0
        
    except Exception as e:
        logger.error(f"Hata: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
