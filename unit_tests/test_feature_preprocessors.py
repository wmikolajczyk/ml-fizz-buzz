import pandas as pd

from feature_preprocessors import (
    PentadecimalFeaturesPreprocessor,
    BinaryFeaturesPreprocessor,
)


def test_pentadecimal_feature_preprocessor():
    X = pd.DataFrame({"number": [0, 1, 2, 3, 4, 5, 10, 15, 23, 30]})
    expected = pd.DataFrame(
        {"number": ["0", "1", "2", "3", "4", "5", "A", "0", "8", "0"]}
    )
    preprocessor = PentadecimalFeaturesPreprocessor()
    transformed = preprocessor.transform(X)
    pd.testing.assert_frame_equal(transformed, expected)


def test_binary_feature_preprocessor():
    X = pd.DataFrame({"number": [0, 1, 2, 3, 4, 5, 10, 15, 23, 30]})
    expected = pd.DataFrame(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 0],
        ],
        dtype=float,
    )
    preprocessor = BinaryFeaturesPreprocessor()
    transformed = preprocessor.transform(X)
    pd.testing.assert_frame_equal(transformed, expected)
