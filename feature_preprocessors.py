from functools import partial

import pandas as pd
from sklearn.base import TransformerMixin

from utils import convert_num_to_given_base


class PentadecimalFeaturesPreprocessor(TransformerMixin):
    def __init__(self):
        self.num_to_pentadecimal_str = partial(convert_num_to_given_base, base=15)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pentadecimal_str = X["number"].apply(self.num_to_pentadecimal_str)
        pentadecimal_last_char = pentadecimal_str.str[-1]
        return pd.DataFrame(pentadecimal_last_char)


class BinaryFeaturesPreprocessor(TransformerMixin):
    def __init__(self):
        self.num_of_last_characters = 8
        self.num_to_binary_str = partial(convert_num_to_given_base, base=2)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        binary_str = X["number"].apply(self.num_to_binary_str)
        last_n_binary_chars = binary_str.str.pad(
            width=self.num_of_last_characters, fillchar="0"
        ).str[-self.num_of_last_characters :]
        # split string to columns
        features = last_n_binary_chars.apply(lambda val: pd.Series(list(val)))
        return features
