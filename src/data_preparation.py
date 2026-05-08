"""
Data Preparation Module — HuggingFace dataset loader for LSTM V2 pipeline.
Only load_from_huggingface() is used by lstm_train_v2.py.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreparation:
    """Minimal data loader for the LSTM V2 training pipeline."""

    def load_from_huggingface(self, hf_config: dict, split: str) -> pd.DataFrame:
        """
        Load a split from a HuggingFace dataset and return it as a DataFrame.

        Parameters
        ----------
        hf_config : dict
            Must contain:
              'repo_id'     — e.g. 'Lightcap/pudu-robot-operation-logs-...'
              'config_name' — e.g. 'partitioned_error_logs'
        split : str
            One of 'train', 'validation', 'test'.

        Returns
        -------
        pd.DataFrame
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HuggingFace 'datasets' library is required. "
                "Install with: pip install datasets"
            ) from exc

        repo_id     = hf_config['repo_id']
        config_name = hf_config['config_name']

        logger.info(
            f"Loading '{split}' split from HuggingFace: {repo_id} [{config_name}]..."
        )
        dataset = load_dataset(repo_id, config_name, split=split)
        df = dataset.to_pandas()
        logger.info(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
