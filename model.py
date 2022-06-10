from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import Pipeline

from dataset import generate_data

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


@lru_cache(maxsize=1)
def get_model(model_path: Path = MODEL_PATH):
    return joblib.load(model_path)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    metrics = {
        "f1": f1_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(y_true, y_pred),
    }
    return metrics


def int_to_pentadecimal_str(num: int) -> str:
    return np.base_repr(num, base=15)


class Preprocessor(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pentadecimal_str = X["number"].apply(int_to_pentadecimal_str)
        pentadecimal_last_chars = pentadecimal_str.apply(lambda val: val[-1])
        features = pd.get_dummies(pentadecimal_last_chars)
        return features


if __name__ == "__main__":
    train_df = generate_data(101, 820)
    val_df = generate_data(821, 1000)
    test_df = generate_data(1, 100)

    X_train, y_train = train_df[["number"]], train_df["label"]
    X_val, y_val = val_df[["number"]], val_df["label"]
    X_test, y_test = test_df[["number"]], test_df["label"]

    model = Pipeline(
        [
            ("preprocess_features", Preprocessor()),
            ("classifier", RandomForestClassifier(n_jobs=-1, random_state=93)),
        ]
    )
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    metrics = evaluate(y_test, test_preds)
    for metric_name, metric_val in metrics.items():
        print(f"{metric_name}\n{metric_val}")

    # save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
