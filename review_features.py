import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ReviewFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        feature_rows = []

        for text in X:
            # Convert missing or non-str to safe string
            if not isinstance(text, str):
                text = "" if text is None else str(text)

            length_chars = len(text)
            tokens = text.split()
            length_words = len(tokens)

            # avg word length
            if length_words > 0:
                avg_word_length = sum(len(t) for t in tokens) / length_words
            else:
                avg_word_length = 0.0

            # punctuation counts
            num_exclamation = text.count("!")
            num_question = text.count("?")
            num_period = text.count(".")
            num_comma = text.count(",")

            # uppercase/lowercase/digit/punct ratios
            num_upper = sum(1 for ch in text if ch.isupper())
            num_lower = sum(1 for ch in text if ch.islower())
            num_digits = sum(1 for ch in text if ch.isdigit())
            num_punct = sum(1 for ch in text if ch in "!?.,;:\"'")

            if (num_upper + num_lower) > 0:
                upper_ratio = num_upper / (num_upper + num_lower)
            else:
                upper_ratio = 0.0

            denom = length_chars if length_chars > 0 else 1
            digit_ratio = num_digits / denom
            punct_ratio = num_punct / denom

            feature_rows.append([
                float(length_chars),
                float(length_words),
                float(avg_word_length),
                float(num_exclamation),
                float(num_question),
                float(num_period),
                float(num_comma),
                float(upper_ratio),
                float(digit_ratio),
                float(punct_ratio),
            ])

        return np.array(feature_rows, dtype=float)
