#!/usr/bin/env python3
"""
LSTM V2 Training Pipeline — Multi-Output Predictive Maintenance
Outputs:
  1. is_failure_now      — Is the robot currently failing?
  2. severity_class      — Event / Warning / Error / Fatal
  3. future_failure_prob — Will it fail within the next 7 days?
  4. hours_to_failure    — Estimated hours until next failure (0–168)
"""

import sys
import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import LSTM_V2_CONFIG, HUGGINGFACE_CONFIG
from data_preparation import DataPreparation
from lstm_models_v2 import MultiOutputLSTMModel, SEVERITY_LABELS, FUTURE_WINDOW_HOURS
from sklearn.preprocessing import StandardScaler

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'lstm_v2_training.log'),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

MODEL_DIR     = Path('models/lstm_v2')
SEQUENCE_LEN  = LSTM_V2_CONFIG.get('sequence_length', 10)
FUTURE_WINDOW = LSTM_V2_CONFIG.get('future_window', FUTURE_WINDOW_HOURS)

# Error type → grouped category (0 = unknown)
ERROR_CATEGORY_MAP = {
    # Navigation
    'NavigationStuck': 1, 'NavigationStopAdvance': 1, 'NavigationSpeedLimit': 1,
    'PlanOverTime': 1, 'PlanFailOverTime': 1, 'ReplanError': 1,
    'CanNotReach': 1, 'PoseNotInit': 1, 'ScheduleAlgoError': 1, 'TaskCannotStart': 1,
    # Sensor loss
    'LostLocalization': 2, 'LostIMU': 2, 'LostEncoder': 2, 'LostRGBD': 2,
    'LostBattery': 2, 'LostLidar': 2, 'LostCAN': 2, 'LostCamera': 2,
    'Lostultrasonic': 2, 'Sensor': 2,
    # Motion / actuator
    'MotionError': 3, 'WheelErrorLeft': 3, 'WheelErrorRight': 3,
    'FailEscapeMotorStuck': 3, 'FailEscapeSlip': 3, 'FailEscapeBumperStripTrigger': 3,
    # Power / battery
    'ChargingPlugError': 4, 'LackOfPower': 4, 'batteryConnectError': 4, 'charge': 4,
    # Map / state
    'MapConfigError': 5, 'SelfStateMonitor': 5, 'FallDropV3': 5,
    'Costmap': 5, 'StixelDetect': 5,
    # Cleaning subsystem
    'BrushError': 6, 'DustAbsorptionError': 6, 'DustMopError': 6,
    'DustReductionError': 6, 'ErrorOfAddWater': 6, 'ErrorOfDrainSewage': 6,
    'FullOfSewage': 6, 'LackOfWater': 6, 'FlowMeterError': 6,
    'RightEdgeSwipeError': 6, 'UndulateError': 6,
    # Hardware / communication
    'CommunicationError': 7, 'InternalError': 7, 'ReflectorDropError': 7,
    'CoverPlate': 7, 'Filter': 7, 'Handrail': 7, 'Screen': 7,
    'FailedToCall': 7, 'FailedToEnter': 7,
}

FEATURE_COLUMNS = [
    'error_count',              # hourly error count
    'task_hour_num',            # hour of day  (0–23)
    'day_of_month',             # day of month (1–31)
    'day_of_week',              # day of week  (0=Monday)
    'robot_id_length',          # len(robot_id string)
    'software_version_length',  # len(soft_version string)
    'product_code_type',        # robot model code  (1–5)
    'hourly_error_rate',        # hourly_ratio
    'error_category',           # grouped error_type (0–7)
]

SEVERITY_MAP = {'Event': 0, 'Warning': 1, 'Error': 2, 'Fatal': 3, 'Critical': 2}
FAILURE_LEVELS = {'Error', 'Fatal', 'Critical'}


# ═════════════════════════════════════════════════════════════════════════════
class LSTMTrainerV2:
    """7-step multi-output LSTM training pipeline."""

    def __init__(self):
        self.data_prep  = DataPreparation()
        self.scaler     = StandardScaler()
        self.lstm_model = None
        self.train_df   = None
        self.val_df     = None
        self.test_df    = None

    # ── Step 1: Load data ────────────────────────────────────────────────────
    def load_data(self):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 1: LOAD DATA FROM HUGGINGFACE")
        logger.info("=" * 70)

        self.train_df = self.data_prep.load_from_huggingface(
            HUGGINGFACE_CONFIG, split='train')
        logger.info(f"  train  : {self.train_df.shape}")

        self.val_df = self.data_prep.load_from_huggingface(
            HUGGINGFACE_CONFIG, split='validation')
        logger.info(f"  val    : {self.val_df.shape}")

        self.test_df = self.data_prep.load_from_huggingface(
            HUGGINGFACE_CONFIG, split='test')
        logger.info(f"  test   : {self.test_df.shape}")

        total = len(self.train_df) + len(self.val_df) + len(self.test_df)
        logger.info(f"  total  : {total:,}")

    # ── Step 2: Feature engineering ──────────────────────────────────────────
    def engineer_features(self, df):
        """Add engineered features and all 4 targets to the dataframe."""
        df = df.copy()

        # Time features
        df['task_time'] = pd.to_datetime(df['task_time'])
        df['day_of_month'] = df['task_time'].dt.day
        df['day_of_week']  = df['task_time'].dt.dayofweek

        df['task_hour']     = pd.to_datetime(df['task_hour'])
        df['task_hour_num'] = df['task_hour'].dt.hour

        # Rename existing columns
        df['error_count']       = df['hourly_error_count']
        df['hourly_error_rate'] = df['hourly_ratio']

        # Identity features
        df['robot_id_length']         = df['robot_id'].astype(str).str.len()
        df['software_version_length'] = df['soft_version'].astype(str).str.len()

        # Product code encoding
        product_map = {'PuduBot2': 1, 'KettyBot': 2, 'Bellabot': 3,
                       'BellaBotPro': 3, 'CC': 4, 'CC1': 4}
        df['product_code_type'] = (
            df['product_code'].map(product_map).fillna(5).astype(int)
        )

        # Error category (from error_type — not derived from error_level)
        df['error_category'] = (
            df['error_type'].map(ERROR_CATEGORY_MAP).fillna(0).astype(int)
        )

        # Target 1: current failure (binary)
        df['is_failure_now'] = (
            df['error_level'].isin(FAILURE_LEVELS).astype(int)
        )

        # Target 2: severity class (0–3)
        df['severity_class'] = (
            df['error_level'].map(SEVERITY_MAP).fillna(0).astype(int)
        )

        # Protect against NaN/Inf
        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)

        return df

    # ── Step 3: Future targets (per-robot look-ahead) ────────────────────────
    def compute_future_targets(self, df):
        """
        For each row, compute:
          future_failure_prob  — binary: any failure within next FUTURE_WINDOW hours
          hours_to_failure_norm — normalized hours to next failure [0, 1]
                                  (0 = imminent, 1 = no failure in window)
        Computed per robot to avoid cross-robot leakage.
        """
        logger.info(f"  Computing future targets (look-ahead: {FUTURE_WINDOW} hours)...")

        robot_groups = []
        for robot_id, rdf in df.groupby('robot_id'):
            rdf = rdf.sort_values('task_time').copy().reset_index(drop=True)
            rdf = self._compute_future_for_robot(rdf)
            robot_groups.append(rdf)

        result = pd.concat(robot_groups, ignore_index=True)
        pos_future = result['future_failure_prob'].sum()
        logger.info(
            f"  future_failure_prob positives: {int(pos_future):,} "
            f"({pos_future / len(result) * 100:.1f}%)"
        )
        return result

    def _compute_future_for_robot(self, df):
        """Vectorised look-ahead for a single robot's sorted dataframe."""
        is_failure = df['error_level'].isin(FAILURE_LEVELS).values
        times      = df['task_time'].values.astype('datetime64[ns]').astype(np.int64)
        window_ns  = int(FUTURE_WINDOW * 3600 * 1e9)

        future_prob  = np.zeros(len(df), dtype=np.float32)
        hours_to_fail = np.full(len(df), float(FUTURE_WINDOW), dtype=np.float32)

        failure_positions = np.where(is_failure)[0]
        for fp in failure_positions:
            fp_time    = times[fp]
            start_time = fp_time - window_ns
            left       = int(np.searchsorted(times, start_time, side='left'))
            for i in range(left, fp):
                future_prob[i] = 1.0
                hrs = (fp_time - times[i]) / 1e9 / 3600
                if hrs < hours_to_fail[i]:
                    hours_to_fail[i] = hrs

        df['future_failure_prob']   = future_prob
        df['hours_to_failure_norm'] = np.clip(hours_to_fail / FUTURE_WINDOW, 0.0, 1.0)
        return df

    # ── Step 4: Sequence creation ─────────────────────────────────────────────
    def create_sequences(self, df):
        """
        Sliding-window sequences → (X, y_dict).

        Label is taken from the step AFTER the sequence (t+1), not the last step
        of the sequence (t).  This avoids leakage: features at step t (e.g.
        error_count, error_category) are correlated with is_failure_now at t
        because both come from the same raw log row.  Shifting the label one
        step forward means the model must genuinely predict the next state from
        prior observations only.
        """
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                logger.warning(f"  Feature '{col}' missing → filled with 0")
                df[col] = 0

        X, y_failure, y_severity, y_future, y_time = [], [], [], [], []

        n = len(df)
        # Range stops one earlier so label index (i + SEQUENCE_LEN) stays in bounds
        for i in range(n - SEQUENCE_LEN):
            seq  = df[FEATURE_COLUMNS].iloc[i: i + SEQUENCE_LEN].values
            last = i + SEQUENCE_LEN          # next-step label (t+1, not in seq)
            X.append(seq)
            y_failure.append(df['is_failure_now'].iloc[last])
            y_severity.append(df['severity_class'].iloc[last])
            y_future.append(df['future_failure_prob'].iloc[last])
            y_time.append(df['hours_to_failure_norm'].iloc[last])

        X = np.nan_to_num(
            np.array(X, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6
        )
        y_dict = {
            'is_failure_now':        np.array(y_failure,  dtype=np.int32),
            'severity_class':        np.array(y_severity, dtype=np.int32),
            'future_failure_prob':   np.array(y_future,   dtype=np.float32),
            'hours_to_failure_norm': np.array(y_time,     dtype=np.float32),
        }
        logger.info(
            f"  Sequences: {X.shape[0]:,}  |  shape per seq: {X.shape[1:]}"
        )
        return X, y_dict

    # ── Step 5: Prepare & normalize ──────────────────────────────────────────
    def prepare_data(self):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2-5: FEATURE ENGINEERING + NORMALIZATION")
        logger.info("=" * 70)

        logger.info("Engineering features...")
        train_df = self.engineer_features(self.train_df)
        val_df   = self.engineer_features(self.val_df)
        test_df  = self.engineer_features(self.test_df)

        logger.info("Computing future targets...")
        train_df = self.compute_future_targets(train_df)
        val_df   = self.compute_future_targets(val_df)
        test_df  = self.compute_future_targets(test_df)

        logger.info("Creating sequences...")
        X_train, y_train = self.create_sequences(train_df)
        X_val,   y_val   = self.create_sequences(val_df)
        X_test,  y_test  = self.create_sequences(test_df)

        # Fit scaler on training features only
        logger.info("Fitting StandardScaler...")
        n, ts, feat = X_train.shape
        X_train_2d  = X_train.reshape(-1, feat)
        self.scaler.fit(X_train_2d)

        def scale(X):
            s = X.shape
            return np.nan_to_num(
                self.scaler.transform(X.reshape(-1, s[-1])).reshape(s),
                nan=0.0, posinf=6.0, neginf=-6.0,
            )

        X_train = scale(X_train)
        X_val   = scale(X_val)
        X_test  = scale(X_test)

        # Class weights for failure imbalance
        n0 = int((y_train['is_failure_now'] == 0).sum())
        n1 = int((y_train['is_failure_now'] == 1).sum())
        w1 = n0 / max(n1, 1)
        failure_class_weight = {0: 1.0, 1: w1}
        logger.info(
            f"  Failure class distribution: normal={n0:,}  failure={n1:,}  "
            f"weight[1]={w1:.2f}"
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, failure_class_weight

    # ── Step 6: Build model ───────────────────────────────────────────────────
    def build_model(self, input_shape):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: BUILD MODEL")
        logger.info("=" * 70)
        self.lstm_model = MultiOutputLSTMModel(LSTM_V2_CONFIG)
        self.lstm_model.build_model(input_shape, n_severity_classes=4)
        self.lstm_model.model.summary(print_fn=logger.info)
        return self.lstm_model

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    def train_model(self, X_train, y_train, X_val, y_val, failure_class_weight):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: TRAIN MODEL")
        logger.info("=" * 70)
        self.lstm_model.train(
            X_train, y_train, X_val, y_val,
            failure_class_weight=failure_class_weight,
        )

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    def evaluate_model(self, X_test, y_test):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 8: EVALUATE MODEL ON TEST SET")
        logger.info("=" * 70)
        metrics = self.lstm_model.evaluate(X_test, y_test, FUTURE_WINDOW)

        logger.info("\n--- Current Failure (Head 1) ---")
        f = metrics['failure']
        logger.info(f"  Accuracy : {f['accuracy']:.4f}")
        logger.info(f"  Precision: {f['precision']:.4f}")
        logger.info(f"  Recall   : {f['recall']:.4f}")
        logger.info(f"  F1-Score : {f['f1']:.4f}")
        logger.info(f"  AUC-ROC  : {f['auc_roc']:.4f}")
        logger.info(f"  TP={f['tp']}  TN={f['tn']}  FP={f['fp']}  FN={f['fn']}")

        logger.info("\n--- Severity Class (Head 2) ---")
        logger.info(f"  Accuracy : {metrics['severity']['accuracy']:.4f}")
        logger.info(metrics['severity']['report'])

        logger.info("\n--- Future Failure 7-day (Head 3) ---")
        fut = metrics['future']
        logger.info(f"  Accuracy : {fut['accuracy']:.4f}")
        logger.info(f"  Precision: {fut['precision']:.4f}")
        logger.info(f"  Recall   : {fut['recall']:.4f}")
        logger.info(f"  F1-Score : {fut['f1']:.4f}")
        logger.info(f"  AUC-ROC  : {fut['auc_roc']:.4f}")

        logger.info("\n--- Time to Failure (Head 4) ---")
        ttf = metrics['time_to_failure']
        logger.info(f"  MAE (normalized): {ttf['mae_normalized']:.4f}")
        logger.info(f"  MAE (hours)     : {ttf['mae_hours']:.1f} h")

        return metrics

    # ── Step 9: Save model & scaler ──────────────────────────────────────────
    def save_model(self):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 9: SAVE MODEL")
        logger.info("=" * 70)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.lstm_model.save_model(MODEL_DIR)
        joblib.dump(self.scaler, MODEL_DIR / 'lstm_v2_scaler.pkl')
        logger.info(f"  Scaler saved → {MODEL_DIR / 'lstm_v2_scaler.pkl'}")

    # ── Step 10: Generate report ──────────────────────────────────────────────
    def generate_report(self, metrics):
        logger.info("\n" + "=" * 70)
        logger.info("STEP 10: GENERATE REPORT")
        logger.info("=" * 70)
        report_path = LOG_DIR / 'lstm_v2_training_report.txt'
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            f"LSTM V2 Multi-Output Training Report",
            f"Generated: {ts}",
            f"Future window: {FUTURE_WINDOW} hours (7 days)",
            "",
            "=== HEAD 1: Current Failure ===",
            f"  Accuracy : {metrics['failure']['accuracy']:.4f}",
            f"  Precision: {metrics['failure']['precision']:.4f}",
            f"  Recall   : {metrics['failure']['recall']:.4f}",
            f"  F1-Score : {metrics['failure']['f1']:.4f}",
            f"  AUC-ROC  : {metrics['failure']['auc_roc']:.4f}",
            f"  TP={metrics['failure']['tp']}  TN={metrics['failure']['tn']}"
            f"  FP={metrics['failure']['fp']}  FN={metrics['failure']['fn']}",
            "",
            "=== HEAD 2: Severity Class ===",
            f"  Accuracy : {metrics['severity']['accuracy']:.4f}",
            metrics['severity']['report'],
            "",
            "=== HEAD 3: Future Failure (7 days) ===",
            f"  Accuracy : {metrics['future']['accuracy']:.4f}",
            f"  Precision: {metrics['future']['precision']:.4f}",
            f"  Recall   : {metrics['future']['recall']:.4f}",
            f"  F1-Score : {metrics['future']['f1']:.4f}",
            f"  AUC-ROC  : {metrics['future']['auc_roc']:.4f}",
            "",
            "=== HEAD 4: Time to Failure ===",
            f"  MAE (normalized): {metrics['time_to_failure']['mae_normalized']:.4f}",
            f"  MAE (hours)     : {metrics['time_to_failure']['mae_hours']:.1f} h",
        ]
        report_path.write_text('\n'.join(lines), encoding='utf-8')
        logger.info(f"  Report saved → {report_path}")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("\n" + "=" * 70)
    logger.info("LSTM V2 MULTI-OUTPUT PREDICTIVE MAINTENANCE — TRAINING START")
    logger.info("=" * 70)

    trainer = LSTMTrainerV2()

    # 1. Load
    trainer.load_data()

    # 2–5. Feature engineering + normalization
    X_train, y_train, X_val, y_val, X_test, y_test, fcw = trainer.prepare_data()

    # 6. Build
    input_shape = (X_train.shape[1], X_train.shape[2])
    trainer.build_model(input_shape)

    # 7. Train
    trainer.train_model(X_train, y_train, X_val, y_val, fcw)

    # 8. Evaluate
    metrics = trainer.evaluate_model(X_test, y_test)

    # 9. Save
    trainer.save_model()

    # 10. Report
    trainer.generate_report(metrics)

    logger.info("\n Training complete.")


if __name__ == '__main__':
    main()
