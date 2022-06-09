from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

from dataset import generate_data


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    metrics = {"f1": f1_score(y_true, y_pred, average="macro")}
    return metrics


def get_last_pentadecimal_char(num: int) -> str:
    return np.base_repr(num, base=15)[-1]


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    pentadecimal_last_chars = df["number"].apply(get_last_pentadecimal_char)
    features = pd.get_dummies(pentadecimal_last_chars)
    return features


if __name__ == "__main__":
    train_df = generate_data(101, 820)
    val_df = generate_data(821, 1000)
    test_df = generate_data(1, 100)

    X_train, y_train = train_df[["number"]], train_df["label"]
    X_val, y_val = val_df[["number"]], val_df["label"]
    X_test, y_test = test_df[["number"]], test_df["label"]

    model = RandomForestClassifier(n_jobs=-1, random_state=93)
    model.fit(process_features(X_train), y_train)

    test_preds = model.predict(process_features(X_test))
    metrics = evaluate(y_test, test_preds)
    print(metrics)
    print(classification_report(y_test, test_preds))
