from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dataset import generate_data
from feature_preprocessors import (
    PentadecimalFeaturesPreprocessor,
    BinaryFeaturesPreprocessor,
)
from utils import evaluate

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "random_forest.joblib"


@lru_cache(maxsize=1)
def get_model(model_path: Path = MODEL_PATH) -> Pipeline:
    return joblib.load(model_path)


def pentadecimal_features_random_forest() -> Pipeline:
    return Pipeline(
        [
            (
                "get_last_char_of_pentadecimal_representation",
                PentadecimalFeaturesPreprocessor(),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    categories=[
                        [
                            "0",
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                            "A",
                            "B",
                            "C",
                            "D",
                            "E",
                        ]
                    ]
                ),
            ),
            ("classifier", RandomForestClassifier(n_jobs=-1, random_state=93)),
        ]
    )


def binary_features_random_forest() -> Pipeline:
    return Pipeline(
        [
            ("preprocess_features", BinaryFeaturesPreprocessor()),
            ("classifier", RandomForestClassifier(n_jobs=-1, random_state=93)),
        ]
    )


if __name__ == "__main__":
    train_df = generate_data(101, 820)
    val_df = generate_data(821, 1000)
    test_df = generate_data(1, 100)

    X_train, y_train = train_df[["number"]], train_df["label"]
    X_val, y_val = val_df[["number"]], val_df["label"]
    X_test, y_test = test_df[["number"]], test_df["label"]

    model = pentadecimal_features_random_forest()
    model.fit(X_train, y_train)

    test_preds = model.predict(X_test)
    metrics = evaluate(y_test, test_preds)
    for metric_name, metric_val in metrics.items():
        print(f"{metric_name}\n{metric_val}")

    # save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
