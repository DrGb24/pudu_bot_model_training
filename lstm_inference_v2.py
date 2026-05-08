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
import pandas as pd
import joblib
from pathlib import Path
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from lstm_models_v2 import (
    MultiOutputLSTMModel, SEVERITY_LABELS, SEVERITY_LABELS_TR, FUTURE_WINDOW_HOURS
)

# Error type → category mapping (same as training)
ERROR_CATEGORY_MAP = {
    'NavigationStuck': 1, 'NavigationStopAdvance': 1, 'NavigationSpeedLimit': 1,
    'PlanOverTime': 1, 'PlanFailOverTime': 1, 'ReplanError': 1,
    'CanNotReach': 1, 'PoseNotInit': 1, 'ScheduleAlgoError': 1, 'TaskCannotStart': 1,
    'LostLocalization': 2, 'LostIMU': 2, 'LostEncoder': 2, 'LostRGBD': 2,
    'LostBattery': 2, 'LostLidar': 2, 'LostCAN': 2, 'LostCamera': 2,
    'Lostultrasonic': 2, 'Sensor': 2,
    'MotionError': 3, 'WheelErrorLeft': 3, 'WheelErrorRight': 3,
    'FailEscapeMotorStuck': 3, 'FailEscapeSlip': 3, 'FailEscapeBumperStripTrigger': 3,
    'ChargingPlugError': 4, 'LackOfPower': 4, 'batteryConnectError': 4, 'charge': 4,
    'MapConfigError': 5, 'SelfStateMonitor': 5, 'FallDropV3': 5, 'Costmap': 5, 'StixelDetect': 5,
    'BrushError': 6, 'DustAbsorptionError': 6, 'DustMopError': 6, 'DustReductionError': 6,
    'ErrorOfAddWater': 6, 'ErrorOfDrainSewage': 6, 'FullOfSewage': 6, 'LackOfWater': 6,
    'FlowMeterError': 6, 'RightEdgeSwipeError': 6, 'UndulateError': 6,
    'CommunicationError': 7, 'InternalError': 7, 'ReflectorDropError': 7,
    'CoverPlate': 7, 'Filter': 7, 'Handrail': 7, 'Screen': 7,
    'FailedToCall': 7, 'FailedToEnter': 7,
}

ERROR_CATEGORY_LABELS = {
    0: 'Bilinmiyor', 1: 'Navigasyon', 2: 'Sensör Kaybı', 3: 'Hareket',
    4: 'Güç/Batarya', 5: 'Harita/Durum', 6: 'Temizlik', 7: 'Donanım/İletişim',
}

PRODUCT_MAP   = {'PuduBot2': 1, 'KettyBot': 2, 'Bellabot': 3, 'BellaBotPro': 3, 'CC': 4, 'CC1': 4}
SEVERITY_MAP  = {'Event': 0, 'Warning': 1, 'Error': 2, 'Fatal': 3, 'Critical': 2}
FAILURE_LEVELS = {'Error', 'Fatal', 'Critical'}

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
        self.lstm_model.model.load_weights(str(model_dir / 'lstm_v2_weights.weights.h5'))

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

    # ── Robot-level inference ─────────────────────────────────────────────────
    def _engineer_robot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering for a single robot's DataFrame (mirrors training)."""
        df = df.copy()
        df['task_time']  = pd.to_datetime(df['task_time'])
        df['task_hour']  = pd.to_datetime(df['task_hour'])
        df = df.sort_values('task_time').reset_index(drop=True)

        df['task_hour_num']           = df['task_hour'].dt.hour
        df['day_of_month']            = df['task_time'].dt.day
        df['day_of_week']             = df['task_time'].dt.dayofweek
        df['error_count']             = df['hourly_error_count']
        df['hourly_error_rate']       = df['hourly_ratio']
        df['robot_id_length']         = df['robot_id'].astype(str).str.len()
        df['software_version_length'] = df['soft_version'].astype(str).str.len()
        df['product_code_type']       = df['product_code'].map(PRODUCT_MAP).fillna(5).astype(int)
        df['error_category']          = df['error_type'].map(ERROR_CATEGORY_MAP).fillna(0).astype(int)

        for col in FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = df[col].replace([np.inf, -np.inf], 0)

        return df

    def _make_sequences(self, df: pd.DataFrame):
        """Sliding window → (sequences array, last-step raw rows)."""
        seq_len = self.input_shape[0]
        n = len(df)
        if n < seq_len:
            # Pad with first row if not enough data
            pad = pd.concat([df.iloc[[0]] * (seq_len - n), df], ignore_index=True)
            X = pad[FEATURE_COLUMNS].values[np.newaxis, ...]
            last_rows = [df.iloc[-1]]
        else:
            X = np.stack([
                df[FEATURE_COLUMNS].values[i: i + seq_len]
                for i in range(n - seq_len + 1)
            ])
            last_rows = [df.iloc[i + seq_len - 1] for i in range(n - seq_len + 1)]
        return X.astype(np.float32), last_rows

    def predict_for_robot(self, robot_id: str, robot_df: pd.DataFrame) -> dict:
        """
        Given a robot's log DataFrame, return aggregated predictions.

        Parameters
        ----------
        robot_id  : str — robot identifier (for reporting)
        robot_df  : pd.DataFrame — raw log rows for this robot
                    Must contain: task_time, task_hour, hourly_error_count,
                    hourly_ratio, robot_id, soft_version, product_code,
                    error_type, error_level

        Returns
        -------
        dict with per-robot aggregated predictions
        """
        df = self._engineer_robot_features(robot_df)
        X, last_rows = self._make_sequences(df)

        # Scale
        s = X.shape
        scaled = self.scaler.transform(X.reshape(-1, s[-1])).reshape(s).astype(np.float32)
        scaled = np.nan_to_num(scaled, nan=0.0, posinf=6.0, neginf=-6.0)

        preds = self.lstm_model.model.predict(scaled, verbose=0)
        p_failures   = preds[0].flatten()
        p_severities = preds[1]            # (N, 4)
        p_futures    = preds[2].flatten()
        p_times      = preds[3].flatten()  # normalized

        # ── HEAD 1: current state = last sequence prediction ──────────────
        p_now     = float(p_failures[-1])
        is_fail   = p_now >= 0.5

        # Aktif hata tipleri (son N satırdan gerçek error_level bazlı)
        recent_df      = df.tail(self.input_shape[0])
        active_errors  = recent_df[recent_df['error_level'].isin(FAILURE_LEVELS)]['error_type'].unique().tolist()
        active_cats    = [ERROR_CATEGORY_LABELS.get(
            int(ERROR_CATEGORY_MAP.get(e, 0)), 'Bilinmiyor') for e in active_errors]
        active_cats    = list(dict.fromkeys(active_cats))  # deduplicate

        # ── HEAD 2: severity = last sequence ──────────────────────────────
        sev_idx    = int(np.argmax(p_severities[-1]))
        sev_label  = SEVERITY_LABELS.get(sev_idx, 'Unknown')
        sev_tr     = SEVERITY_LABELS_TR.get(sev_idx, 'Bilinmiyor')
        sev_scores = p_severities[-1].tolist()  # probabilities per class

        # ── HEAD 3: monthly repair probability = max future_prob over all seqs ──
        p_future_max = float(np.max(p_futures))
        p_future_now = float(p_futures[-1])

        # ── HEAD 4: soonest predicted failure (minimum hours) ────────────────
        min_time_norm = float(np.min(p_times))
        est_hours     = round(min_time_norm * self.future_window, 1)
        est_days      = round(est_hours / 24, 2)

        risk = self._risk_level(p_now, p_future_max)

        return {
            'robot_id': robot_id,
            # Head 1
            'is_failure_now':    is_fail,
            'failure_prob_now':  round(p_now, 4),
            'active_error_types': active_errors,
            'active_error_categories': active_cats,
            # Head 2
            'severity_now':      sev_label,
            'severity_now_tr':   sev_tr,
            'severity_score':    sev_idx,
            'severity_probs':    {SEVERITY_LABELS[i]: round(sev_scores[i], 3) for i in range(4)},
            # Head 3
            'monthly_repair_prob': round(p_future_max, 4),
            'next_7d_fail_prob':   round(p_future_now, 4),
            # Head 4
            'est_hours_to_failure': est_hours,
            'est_days_to_failure':  est_days,
            # Summary
            'risk_level': risk,
        }

    def robot_report(self, robot_id: str, robot_df: pd.DataFrame) -> str:
        """
        Generate a human-readable Turkish report for a single robot.
        """
        r   = self.predict_for_robot(robot_id, robot_df)
        now = datetime.now().strftime('%Y-%m-%d %H:%M')

        durum    = 'ARIZALI ⚠️' if r['is_failure_now'] else 'NORMAL ✓'
        hata_str = ', '.join(r['active_error_types']) if r['active_error_types'] else 'Yok'
        kat_str  = ', '.join(r['active_error_categories']) if r['active_error_categories'] else 'Yok'

        sev_bar = ' | '.join(
            f"{k}: %{v*100:.0f}" for k, v in r['severity_probs'].items()
        )

        aylik = r['monthly_repair_prob'] * 100
        if aylik >= 70:
            aylik_yorum = 'YÜKSEK ihtimalle bakım gerekecek'
        elif aylik >= 40:
            aylik_yorum = 'Bakım gerekebilir, takip edilmeli'
        else:
            aylik_yorum = 'Bakım gereksinimi düşük'

        if r['est_hours_to_failure'] <= 12:
            zaman_yorum = f"⛔ {r['est_hours_to_failure']:.1f} saat içinde hata riski ÇOK YÜKSEK"
        elif r['est_hours_to_failure'] <= 48:
            zaman_yorum = f"⚠️  ~{r['est_hours_to_failure']:.0f} saat içinde hata bekleniyor"
        elif r['est_hours_to_failure'] <= 120:
            zaman_yorum = f"📅 Tahmini {r['est_days_to_failure']:.1f} gün içinde arıza olabilir"
        else:
            zaman_yorum = f"✅ Yakın vadede arıza öngörülmüyor (>{r['est_hours_to_failure']:.0f} saat)"

        lines = [
            f"{'='*60}",
            f"  ROBOT DURUM RAPORU — {now}",
            f"  Robot ID : {r['robot_id']}",
            f"  Risk     : {r['risk_level']}",
            f"{'='*60}",
            f"",
            f"  [HEAD 1] ANLÍK DURUM",
            f"  ├─ Durum          : {durum}  (P={r['failure_prob_now']:.1%})",
            f"  ├─ Aktif hatalar  : {hata_str}",
            f"  └─ Hata kategorisi: {kat_str}",
            f"",
            f"  [HEAD 2] ARIZA ŞİDDETİ",
            f"  ├─ Mevcut şiddet  : {r['severity_now_tr']} ({r['severity_now']})",
            f"  └─ Dağılım        : {sev_bar}",
            f"",
            f"  [HEAD 3] AYLIK TAMİR GEREKSİNİMİ",
            f"  ├─ Bu ay bakım ihtimali : %{aylik:.1f}",
            f"  ├─ 7 günlük arıza ihtim.: %{r['next_7d_fail_prob']*100:.1f}",
            f"  └─ Yorum          : {aylik_yorum}",
            f"",
            f"  [HEAD 4] TAHMİNİ ARIZA ZAMANI",
            f"  └─ {zaman_yorum}",
            f"",
            f"{'='*60}",
        ]
        return '\n'.join(lines)

    def fleet_report(self, robot_dfs: dict) -> str:
        """
        Generate reports for multiple robots.

        Parameters
        ----------
        robot_dfs : dict  {robot_id: DataFrame}
        """
        reports = []
        for rid, df in robot_dfs.items():
            reports.append(self.robot_report(rid, df))
        return '\n\n'.join(reports)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    engine = LSTMInferenceV2()

    # Örnek: normal robot log satırları (gerçekte HuggingFace'den ya da DB'den gelir)
    def _make_demo_df(error_level, error_type, error_count, hourly_ratio, n=15):
        now = pd.Timestamp.now()
        rows = []
        for i in range(n):
            t = now - pd.Timedelta(hours=n - i)
            rows.append({
                'task_time':          t,
                'task_hour':          t.replace(minute=0, second=0),
                'hourly_error_count': error_count + (i % 3),
                'hourly_ratio':       hourly_ratio,
                'robot_id':           'ROBOT_DEMO',
                'soft_version':       'v3.1.2',
                'product_code':       'PuduBot2',
                'error_type':         error_type,
                'error_level':        error_level,
            })
        return pd.DataFrame(rows)

    normal_df  = _make_demo_df('Event',   'NavigationStuck', 2,  0.02)
    failing_df = _make_demo_df('Fatal',   'WheelErrorLeft',  45, 0.80)

    print(engine.robot_report('ROBOT-NORMAL-01', normal_df))
    print()
    print(engine.robot_report('ROBOT-ARIZALI-07', failing_df))
