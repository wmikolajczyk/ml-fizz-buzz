import pandas as pd

from utils import evaluate


def test_model_quality(model_for_test):
    X = pd.DataFrame({'number': [0, 1, 2, 3, 4, 5, 10, 15, 23, 30]})
    y = ["FizzBuzz", "", "", "Fizz", "", "Buzz", "Buzz", "FizzBuzz", "", "FizzBuzz"]
    preds = model_for_test.predict(X)
    metrics = evaluate(y, preds)
    assert metrics["f1"] >= 0.8