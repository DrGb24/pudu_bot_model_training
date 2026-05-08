#!/usr/bin/env python3
"""
LSTM V2 Inference Engine — Multi-Output Predictive Maintenance

For each 10-step input sequence, returns:
  is_failure_now      — bool: robot currently failing?
  failure_prob_now    — float [0,1]: probability of current failure
  severity_now        — str: predicted severity label (Event/Warning/Error/Fatal)
  severity_now_tr     — str: Turkish severity label
  severity_score      — int [0,3]: severity class index
  future_failure_prob — float [0,1]: probability of failure in next 7 days
  est_hours_to_failure — float: estimated hours until next failure (0–168)
  est_days_to_failure  — float: same value in days
  risk_level           — str: DUSUK / ORTA / YUKSEK
"""

import json
import logging
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from lstm_models_v2 import (
    MultiOutputLSTMModel, SEVERITY_LABELS, SEVERITY_LABELS_TR, FUTURE_WINDOW_HOURS
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path('models/lstm_v2')

FEATURE_COLUMNS = [
    'error_count', 'task_hour_num', 'day_of_month', 'day_of_week',
    'robot_id_length', 'software_version_length',
    'product_code_type', 'hourly_error_rate', 'error_category',
]


class LSTMInferenceV2:
    """
    Production inference engine for the V2 multi-output LSTM.

    Usage
    -----
    engine = LSTMInferenceV2()
    result = engine.predict(sequence)   # sequence shape: (10, 9)
    """

    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)

        # Load config
        with open(model_dir / 'lstm_v2_config.json') as f:
            meta = json.load(f)

        self.config       = meta['config']
        self.input_shape  = tuple(meta['input_shape'])   # (seq_len, n_features)
        self.future_window = self.config.get('future_window', FUTURE_WINDOW_HOURS)

        # Build architecture and load weights
        self.lstm_model = MultiOutputLSTMModel(self.config)
        self.lstm_model.build_model(self.input_shape, n_severity_classes=4)
        self.lstm_model.model.load_weights(str(model_dir / 'lstm_v2_weights.h5'))

        # Load fitted scaler
        self.scaler = joblib.load(model_dir / 'lstm_v2_scaler.pkl')

        logger.info(
            f"LSTMInferenceV2 ready | input: {self.input_shape} | "
            f"future window: {self.future_window} h"
        )

    # ──────────────────────────────────────────────────────────────────────────
    def predict(self, sequence: np.ndarray) -> dict:
        """
        Predict for a single sequence.

        Parameters
        ----------
        sequence : ndarray  shape (seq_len, n_features) = (10, 9)
                   Features must be in FEATURE_COLUMNS order.

        Returns
        -------
        dict with all 4 model outputs plus derived fields.
        """
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, ...]       # → (1, 10, 9)

        # Scale
        s = sequence.shape
        scaled = self.scaler.transform(
            sequence.reshape(-1, s[-1])
        ).reshape(s).astype(np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)

        # Predict
        preds = self.lstm_model.model.predict(scaled, verbose=0)
        p_failure   = float(preds[0].flatten()[0])
        p_severity  = preds[1][0]                     # shape (4,)
        p_future    = float(preds[2].flatten()[0])
        p_time_norm = float(preds[3].flatten()[0])    # [0,1]

        severity_idx   = int(np.argmax(p_severity))
        severity_label = SEVERITY_LABELS.get(severity_idx, 'Unknown')
        severity_tr    = SEVERITY_LABELS_TR.get(severity_idx, 'Bilinmiyor')

        est_hours = p_time_norm * self.future_window
        est_days  = est_hours / 24.0

        return {
            # Head 1 — current state
            'is_failure_now':        p_failure >= 0.5,
            'failure_prob_now':      round(p_failure, 4),

            # Head 2 — severity
            'severity_now':          severity_label,
            'severity_now_tr':       severity_tr,
            'severity_score':        severity_idx,

            # Head 3 — future
            'future_failure_prob':   round(p_future, 4),

            # Head 4 — time to failure
            'est_hours_to_failure':  round(est_hours, 1),
            'est_days_to_failure':   round(est_days, 2),

            # Derived risk level
            'risk_level':            self._risk_level(p_failure, p_future),
        }

    def predict_batch(self, sequences: np.ndarray) -> list[dict]:
        """
        Predict for multiple sequences at once.

        Parameters
        ----------
        sequences : ndarray  shape (N, seq_len, n_features)

        Returns
        -------
        list of dicts, one per sequence.
        """
        s = sequences.shape
        scaled = self.scaler.transform(
            sequences.reshape(-1, s[-1])
        ).reshape(s).astype(np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)

        preds = self.lstm_model.model.predict(scaled, verbose=0)
        p_failures   = preds[0].flatten()
        p_severities = preds[1]
        p_futures    = preds[2].flatten()
        p_times      = preds[3].flatten()

        results = []
        for i in range(len(sequences)):
            pf  = float(p_failures[i])
            ps  = p_severities[i]
            pfu = float(p_futures[i])
            pt  = float(p_times[i])
            sev = int(np.argmax(ps))
            hrs = pt * self.future_window
            results.append({
                'is_failure_now':       pf >= 0.5,
                'failure_prob_now':     round(pf, 4),
                'severity_now':         SEVERITY_LABELS.get(sev, 'Unknown'),
                'severity_now_tr':      SEVERITY_LABELS_TR.get(sev, 'Bilinmiyor'),
                'severity_score':       sev,
                'future_failure_prob':  round(pfu, 4),
                'est_hours_to_failure': round(hrs, 1),
                'est_days_to_failure':  round(hrs / 24, 2),
                'risk_level':           self._risk_level(pf, pfu),
            })
        return results

    @staticmethod
    def _risk_level(p_now: float, p_future: float) -> str:
        """Combine current and future probabilities into a risk level."""
        combined = max(p_now, p_future * 0.7)
        if combined >= 0.70:
            return 'YUKSEK'
        elif combined >= 0.40:
            return 'ORTA'
        return 'DUSUK'

    def get_model_info(self) -> dict:
        return {
            'input_shape':    self.input_shape,
            'feature_columns': FEATURE_COLUMNS,
            'future_window_hours': self.future_window,
            'outputs': [
                'is_failure_now (binary)',
                'severity_class (Event/Warning/Error/Fatal)',
                f'future_failure_prob (next {self.future_window}h)',
                'hours_to_failure_norm (regression)',
            ],
        }


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    engine = LSTMInferenceV2()

    print("\nModel Info:")
    for k, v in engine.get_model_info().items():
        print(f"  {k}: {v}")

    # Feature order: error_count, task_hour_num, day_of_month, day_of_week,
    #                robot_id_length, software_version_length,
    #                product_code_type, hourly_error_rate, error_category
    normal_seq = np.array([
        [2,  10, 1, 0, 8, 12, 1, 0.02, 1],
        [1,   9, 1, 0, 8, 12, 1, 0.01, 1],
        [3,  11, 1, 0, 8, 12, 1, 0.03, 1],
        [2,  10, 2, 0, 8, 12, 1, 0.02, 0],
        [1,   8, 2, 0, 8, 12, 1, 0.01, 0],
        [2,  12, 2, 0, 8, 12, 1, 0.02, 1],
        [3,  13, 3, 1, 8, 12, 1, 0.03, 1],
        [2,  14, 3, 1, 8, 12, 1, 0.02, 0],
        [1,  15, 3, 1, 8, 12, 1, 0.01, 0],
        [2,  16, 3, 1, 8, 12, 1, 0.02, 1],
    ], dtype=np.float32)

    failing_seq = np.array([
        [25, 22, 1, 0, 8, 12, 1, 0.45, 3],
        [30, 23, 1, 0, 8, 12, 1, 0.52, 2],
        [18, 21, 1, 0, 8, 12, 1, 0.38, 3],
        [22, 22, 2, 0, 8, 12, 1, 0.41, 2],
        [35, 23, 2, 0, 8, 12, 1, 0.60, 4],
        [28, 20, 2, 0, 8, 12, 1, 0.49, 3],
        [40, 22, 3, 1, 8, 12, 1, 0.70, 2],
        [45, 23, 3, 1, 8, 12, 1, 0.78, 4],
        [38, 21, 3, 1, 8, 12, 1, 0.65, 3],
        [50, 22, 3, 1, 8, 12, 1, 0.85, 2],
    ], dtype=np.float32)

    for label, seq in [('NORMAL Robot', normal_seq), ('ARIZALI Robot', failing_seq)]:
        r = engine.predict(seq)
        print(f"\n── {label} ──────────────────────────────")
        print(f"  Şu an arızalı?         : {'EVET' if r['is_failure_now'] else 'HAYIR'} "
              f"(P={r['failure_prob_now']:.4f})")
        print(f"  Arıza şiddeti          : {r['severity_now_tr']} ({r['severity_now']})")
        print(f"  7 günlük arıza olasılığı: {r['future_failure_prob']:.4f}")
        print(f"  Tahmini arızaya kalan  : {r['est_hours_to_failure']:.1f} saat "
              f"({r['est_days_to_failure']:.1f} gün)")
        print(f"  Risk seviyesi          : {r['risk_level']}")
