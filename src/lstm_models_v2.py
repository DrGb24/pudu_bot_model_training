"""
Multi-Output LSTM Model for Predictive Maintenance V2

Outputs:
  1. is_failure_now      - binary: robot currently failing? (Error/Fatal)
  2. severity_class      - 4-class: Event=0, Warning=1, Error=2, Fatal=3
  3. future_failure_prob - binary: will it fail within 7 days?
  4. hours_to_failure_norm - regression [0,1]: normalized time to next failure
                             (0 = imminent, 1 = no failure within window)
"""

import json
import logging
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, mean_absolute_error,
)

logger = logging.getLogger(__name__)

SEVERITY_LABELS    = {0: 'Event',  1: 'Warning',  2: 'Error',  3: 'Fatal'}
SEVERITY_LABELS_TR = {0: 'Bilgi',  1: 'Uyarı',    2: 'Hata',   3: 'Kritik'}
FUTURE_WINDOW_HOURS = 168  # 7 days


class MultiOutputLSTMModel:
    """
    Multi-output LSTM built with the Keras Functional API.
    Shared LSTM backbone feeds into 4 independent prediction heads.
    """

    def __init__(self, config):
        self.config  = config
        self.model   = None
        self.history = None

    # ──────────────────────────────────────────────────────────────────────
    def build_model(self, input_shape, n_severity_classes=4):
        """Build and compile the multi-output model."""
        lstm_units   = self.config.get('lstm_units',    128)
        dropout      = self.config.get('dropout_rate',  0.3)
        dense_units  = self.config.get('dense_units',   64)
        lr           = self.config.get('learning_rate', 0.0001)

        inputs = keras.Input(shape=input_shape, name='input')

        # Shared LSTM backbone
        x = layers.LSTM(lstm_units, activation='tanh',
                         return_sequences=True, name='lstm_1')(inputs)
        x = layers.Dropout(dropout, name='drop_1')(x)
        x = layers.LSTM(lstm_units // 2, activation='tanh',
                         return_sequences=False, name='lstm_2')(x)
        x = layers.Dropout(dropout, name='drop_2')(x)
        shared = layers.Dense(dense_units, activation='relu',
                               name='shared_dense')(x)

        # Head 1 — Current failure (binary)
        h1 = layers.Dense(32, activation='relu', name='h1_dense')(shared)
        out_failure = layers.Dense(1, activation='sigmoid',
                                    name='is_failure_now')(h1)

        # Head 2 — Severity class (softmax 4-class)
        h2 = layers.Dense(32, activation='relu', name='h2_dense')(shared)
        out_severity = layers.Dense(n_severity_classes, activation='softmax',
                                     name='severity_class')(h2)

        # Head 3 — Future failure probability within 7 days (binary)
        h3 = layers.Dense(32, activation='relu', name='h3_dense')(shared)
        out_future = layers.Dense(1, activation='sigmoid',
                                   name='future_failure_prob')(h3)

        # Head 4 — Normalized time to failure (regression)
        h4 = layers.Dense(32, activation='relu', name='h4_dense')(shared)
        out_time = layers.Dense(1, activation='sigmoid',
                                 name='hours_to_failure_norm')(h4)

        self.model = Model(
            inputs=inputs,
            outputs=[out_failure, out_severity, out_future, out_time],
            name='MultiOutputLSTM_V2',
        )

        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss={
                'is_failure_now':        'binary_crossentropy',
                'severity_class':        'sparse_categorical_crossentropy',
                'future_failure_prob':   'binary_crossentropy',
                'hours_to_failure_norm': 'mse',
            },
            loss_weights={
                'is_failure_now':        1.0,
                'severity_class':        0.5,
                'future_failure_prob':   1.0,
                'hours_to_failure_norm': 0.5,
            },
            metrics={
                'is_failure_now':        ['accuracy'],
                'severity_class':        ['accuracy'],
                'future_failure_prob':   ['accuracy'],
                'hours_to_failure_norm': ['mae'],
            },
        )

        logger.info(
            f"MultiOutputLSTM_V2 built — input: {input_shape}, "
            f"params: {self.model.count_params():,}"
        )
        return self.model

    # ──────────────────────────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val, failure_class_weight=None):
        """
        Train the model.

        Parameters
        ----------
        X_train, X_val : ndarray  shape (N, seq_len, n_features)
        y_train, y_val : dict with keys:
            is_failure_now, severity_class, future_failure_prob, hours_to_failure_norm
        failure_class_weight : dict {0: w0, 1: w1}  for imbalanced failure labels
        """
        epochs   = self.config.get('epochs', 100)
        batch_sz = self.config.get('batch_size', 16)
        patience = self.config.get('early_stopping_patience', 15)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                              min_lr=1e-7, verbose=1),
        ]

        # Sample weights based on failure class imbalance (applied to all outputs)
        sample_weight = None
        if failure_class_weight is not None:
            y_f = y_train['is_failure_now']
            w1  = failure_class_weight.get(1, 1.0)
            w0  = failure_class_weight.get(0, 1.0)
            sample_weight = np.where(y_f == 1, w1, w0).astype(np.float32)

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_sz,
            callbacks=callbacks,
            sample_weight=sample_weight,
            verbose=1,
        )
        return self.history

    # ──────────────────────────────────────────────────────────────────────
    def evaluate(self, X_test, y_test, future_window=FUTURE_WINDOW_HOURS):
        """Evaluate all 4 heads and return a metrics dict."""
        preds = self.model.predict(X_test, verbose=0)
        p_failure     = preds[0].flatten()
        p_severity_2d = preds[1]                   # shape (N, 4) — keep 2D
        p_future      = preds[2].flatten()
        p_time        = preds[3].flatten()

        results = {}

        # Head 1 — Current failure
        y_f      = y_test['is_failure_now']
        y_f_pred = (p_failure >= 0.5).astype(int)
        tn, fp_n, fn, tp = confusion_matrix(y_f, y_f_pred).ravel()
        results['failure'] = {
            'accuracy':  float(accuracy_score(y_f, y_f_pred)),
            'precision': float(precision_score(y_f, y_f_pred, zero_division=0)),
            'recall':    float(recall_score(y_f, y_f_pred, zero_division=0)),
            'f1':        float(f1_score(y_f, y_f_pred, zero_division=0)),
            'auc_roc':   float(roc_auc_score(y_f, p_failure)),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp_n), 'fn': int(fn),
        }

        # Head 2 — Severity
        y_s      = y_test['severity_class']
        y_s_pred = np.argmax(p_severity_2d, axis=1)
        results['severity'] = {
            'accuracy': float(accuracy_score(y_s, y_s_pred)),
            'report':   classification_report(
                y_s, y_s_pred,
                target_names=list(SEVERITY_LABELS.values()),
                zero_division=0,
            ),
        }

        # Head 3 — Future failure
        y_fut      = y_test['future_failure_prob']
        y_fut_pred = (p_future >= 0.5).astype(int)
        results['future'] = {
            'accuracy':  float(accuracy_score(y_fut, y_fut_pred)),
            'precision': float(precision_score(y_fut, y_fut_pred, zero_division=0)),
            'recall':    float(recall_score(y_fut, y_fut_pred, zero_division=0)),
            'f1':        float(f1_score(y_fut, y_fut_pred, zero_division=0)),
            'auc_roc':   float(roc_auc_score(y_fut, p_future)),
        }

        # Head 4 — Time to failure
        y_t = y_test['hours_to_failure_norm']
        mae = float(mean_absolute_error(y_t, p_time))
        results['time_to_failure'] = {
            'mae_normalized': mae,
            'mae_hours':      mae * future_window,
        }

        return results

    # ──────────────────────────────────────────────────────────────────────
    def save_model(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_weights(str(model_dir / 'lstm_v2_weights.h5'))

        meta = {
            'config':       self.config,
            'input_shape':  list(self.model.input_shape[1:]),
            'output_names': self.model.output_names,
        }
        with open(model_dir / 'lstm_v2_config.json', 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Model saved → {model_dir}")

    def load_model(self, model_dir, input_shape, n_severity_classes=4):
        model_dir = Path(model_dir)
        with open(model_dir / 'lstm_v2_config.json') as f:
            meta = json.load(f)
        self.config = meta['config']
        self.build_model(tuple(input_shape), n_severity_classes)
        self.model.load_weights(str(model_dir / 'lstm_v2_weights.h5'))
        logger.info("MultiOutputLSTM_V2 weights loaded.")
        return self.model
