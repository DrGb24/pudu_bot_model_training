"""
KPI Metrics Module - Calculate performance metrics based on requirements
Tracks Model Performance, Operational, System, and Financial KPIs
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KPIMetrics:
    """Calculate and track KPIs"""
    
    # Target thresholds from KPI requirements
    THRESHOLDS = {
        'prediction_accuracy': 0.85,      # >= 85%
        'recall': 0.85,                   # >= 85%
        'precision': 0.80,                # >= 80%
        'f1_score': 0.80,                 # >= 0.8
        'false_alarm_rate': 0.10,         # <= 10%
        'system_latency': 60,             # <= 60 seconds
        'system_uptime': 0.99,            # >= 99%
        'error_handling_rate': 0.95,      # >= 95%
        'connectivity_health': 0.95,      # >= 95%
    }
    
    def __init__(self):
        self.kpi_history = []
        self.timestamp = None
        
    def calculate_model_performance_kpis(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate Model Performance KPIs
        
        Returns:
            dict with all model performance metrics
        """
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Prediction Accuracy: (TP + TN) / Total
        accuracy = accuracy_score(y_true, y_pred)
        
        # Recall: TP / (TP + FN) - True Positive Rate
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Precision: TP / (TP + FP)
        precision = precision_score(y_true, y_pred, zero_division=0)
        
        # F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # False Alarm Rate: FP / Total
        false_alarm_rate = fp / len(y_true) if len(y_true) > 0 else 0
        
        # AUC-ROC if probabilities available
        auc_roc = None
        if y_pred_proba is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                auc_roc = None
        
        kpis = {
            'prediction_accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
            'false_alarm_rate': false_alarm_rate,
            'auc_roc': auc_roc,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
        
        # Check thresholds
        kpis['accuracy_meets_target'] = accuracy >= self.THRESHOLDS['prediction_accuracy']
        kpis['recall_meets_target'] = recall >= self.THRESHOLDS['recall']
        kpis['precision_meets_target'] = precision >= self.THRESHOLDS['precision']
        kpis['f1_meets_target'] = f1 >= self.THRESHOLDS['f1_score']
        kpis['false_alarm_meets_target'] = false_alarm_rate <= self.THRESHOLDS['false_alarm_rate']
        
        logger.info(f"Model Performance KPIs calculated:")
        logger.info(f"  Accuracy: {accuracy:.4f} (Target: {self.THRESHOLDS['prediction_accuracy']})")
        logger.info(f"  Recall: {recall:.4f} (Target: {self.THRESHOLDS['recall']})")
        logger.info(f"  Precision: {precision:.4f} (Target: {self.THRESHOLDS['precision']})")
        logger.info(f"  F1-Score: {f1:.4f} (Target: {self.THRESHOLDS['f1_score']})")
        logger.info(f"  False Alarm Rate: {false_alarm_rate:.4f} (Target: <= {self.THRESHOLDS['false_alarm_rate']})")
        
        return kpis
    
    def calculate_operational_kpis(self, failure_data, error_data):
        """
        Calculate Operational KPIs
        
        Parameters:
        - failure_data: DataFrame with failure information (timestamps, types)
        - error_data: DataFrame with error logs
        
        Returns:
            dict with operational metrics
        """
        
        kpis = {}
        
        # MTBF: Mean Time Between Failures
        if len(failure_data) > 1:
            mtbf = self._calculate_mtbf(failure_data)
            kpis['mtbf'] = mtbf
        else:
            kpis['mtbf'] = 0
        
        # Error Rate: Total Errors / Total Time
        total_errors = len(error_data)
        total_time_hours = 24 * 30  # Assuming 30 days
        error_rate = total_errors / total_time_hours if total_time_hours > 0 else 0
        kpis['error_rate'] = error_rate
        
        # Critical Error Rate: Critical Errors / Total Errors
        critical_errors = error_data[error_data['severity'] == 'critical'].shape[0] if 'severity' in error_data.columns else 0
        critical_error_rate = critical_errors / total_errors if total_errors > 0 else 0
        kpis['critical_error_rate'] = critical_error_rate
        kpis['critical_error_rate_meets_target'] = critical_error_rate <= 0.20
        
        # Failure Frequency
        kpis['failure_frequency'] = len(failure_data) / 30  # Per day average
        
        logger.info(f"Operational KPIs calculated:")
        logger.info(f"  MTBF: {kpis.get('mtbf', 0):.2f} hours")
        logger.info(f"  Error Rate: {error_rate:.4f} errors/hour")
        logger.info(f"  Critical Error Rate: {critical_error_rate:.4f} (Target: <= 0.20)")
        
        return kpis
    
    def calculate_system_kpis(self, inference_times, uptime_data, log_counts):
        """
        Calculate System KPIs
        
        Parameters:
        - inference_times: List of model prediction times in seconds
        - uptime_data: Dict with system uptime metrics
        - log_counts: Number of logs processed
        
        Returns:
            dict with system metrics
        """
        
        # System Latency: prediction_time - log_time <= 60 seconds
        avg_latency = np.mean(inference_times) if (inference_times is not None and len(inference_times) > 0) else 0
        kpis = {
            'system_latency': avg_latency,
            'latency_meets_target': avg_latency <= self.THRESHOLDS['system_latency'],
            'avg_inference_time': avg_latency,
        }
        
        # Data Processing Time
        data_processing_time = np.percentile(inference_times, 95) if (inference_times is not None and len(inference_times) > 0) else 0
        kpis['data_processing_time_p95'] = data_processing_time
        kpis['processing_meets_target'] = data_processing_time <= 30
        
        # System Uptime >= 99%
        uptime_percentage = uptime_data.get('uptime_percentage', 0.99)
        kpis['system_uptime'] = uptime_percentage
        kpis['uptime_meets_target'] = uptime_percentage >= self.THRESHOLDS['system_uptime']
        
        # Connectivity Health Score >= 95%
        connectivity_score = uptime_data.get('connectivity_success_rate', 0.95)
        kpis['connectivity_health_score'] = connectivity_score
        kpis['connectivity_meets_target'] = connectivity_score >= self.THRESHOLDS['connectivity_health']
        
        # Log Throughput: logs per second
        throughput = log_counts if log_counts else 0
        kpis['log_throughput'] = throughput
        
        logger.info(f"System KPIs calculated:")
        logger.info(f"  System Latency: {avg_latency:.4f}s (Target: <= {self.THRESHOLDS['system_latency']}s)")
        logger.info(f"  System Uptime: {uptime_percentage:.4f} (Target: >= {self.THRESHOLDS['system_uptime']})")
        logger.info(f"  Connectivity Health: {connectivity_score:.4f} (Target: >= {self.THRESHOLDS['connectivity_health']})")
        
        return kpis
    
    def calculate_financial_kpis(self, avoided_failures, baseline_failures, 
                               cost_per_failure, system_cost, avoided_maintenance_cost):
        """
        Calculate Financial & Business KPIs
        
        Parameters:
        - avoided_failures: Number of failures prevented
        - baseline_failures: Expected failures without system
        - cost_per_failure: Cost per failure
        - system_cost: Total system investment cost
        - avoided_maintenance_cost: Cost savings from maintenance optimization
        
        Returns:
            dict with financial metrics
        """
        
        kpis = {}
        
        # Avoided Failures
        kpis['avoided_failures'] = avoided_failures
        
        # Cost Savings: Avoided Failures × Cost per Failure
        cost_savings = avoided_failures * cost_per_failure
        kpis['cost_savings'] = cost_savings
        
        # ROI: (Cost Savings - System Cost) / System Cost
        roi = (cost_savings - system_cost) / system_cost if system_cost > 0 else 0
        kpis['roi'] = roi
        kpis['roi_positive'] = roi > 0
        
        # Downtime Cost (example calculation)
        downtime_hours = 0  # Should be tracked from actual data
        cost_per_hour = 5000  # Example
        downtime_cost = downtime_hours * cost_per_hour
        kpis['downtime_cost'] = downtime_cost
        
        # Maintenance Cost Optimization
        kpis['maintenance_cost_optimized'] = True  # Predictive maintenance should reduce costs
        
        # Payback period in months
        if cost_savings > 0:
            payback_period = system_cost / (cost_savings / 12)  # Assuming annual savings
        else:
            payback_period = float('inf')
        kpis['payback_period_months'] = payback_period
        
        logger.info(f"Financial KPIs calculated:")
        logger.info(f"  Avoided Failures: {avoided_failures}")
        logger.info(f"  Cost Savings: ${cost_savings:,.2f}")
        logger.info(f"  ROI: {roi:.2%} (Target: Positive)")
        logger.info(f"  Payback Period: {payback_period:.2f} months")
        
        return kpis
    
    def _calculate_mtbf(self, failure_data):
        """Calculate Mean Time Between Failures"""
        if len(failure_data) < 2:
            return 0
        
        # Assuming failure_data has a timestamp column
        times = sorted(failure_data.index)
        intervals = np.diff(times)
        mtbf = np.mean(intervals) if len(intervals) > 0 else 0
        return mtbf
    
    def generate_kpi_report(self, model_kpis, operational_kpis, system_kpis, financial_kpis):
        """Generate comprehensive KPI report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': model_kpis,
            'operational': operational_kpis,
            'system': system_kpis,
            'financial': financial_kpis,
        }
        
        # Overall system health
        all_targets_met = (
            model_kpis.get('accuracy_meets_target', False) and
            model_kpis.get('recall_meets_target', False) and
            system_kpis.get('uptime_meets_target', False) and
            financial_kpis.get('roi_positive', False)
        )
        
        report['system_health'] = 'HEALTHY' if all_targets_met else 'NEEDS ATTENTION'
        
        self.kpi_history.append(report)
        
        return report
    
    def display_kpi_summary(self, report):
        """Display KPI report in console"""
        print("\n" + "="*60)
        print("KPI PERFORMANCE REPORT")
        print("="*60)
        print(f"Timestamp: {report['timestamp']}")
        print(f"System Status: {report['system_health']}")
        
        print("\n--- MODEL PERFORMANCE ---")
        for key, value in report['model_performance'].items():
            if not key.endswith('_target'):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n--- OPERATIONAL ---")
        for key, value in report['operational'].items():
            if not key.endswith('_target'):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n--- SYSTEM ---")
        for key, value in report['system'].items():
            if not key.endswith('_target'):
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print("\n--- FINANCIAL ---")
        for key, value in report['financial'].items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        print("="*60 + "\n")
