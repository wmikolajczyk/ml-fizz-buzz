from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, classification_report


def convert_num_to_given_base(num: int, base: int) -> str:
    return np.base_repr(num, base=base)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    metrics = {
        "f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics
