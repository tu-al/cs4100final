# length_weighted_logreg.py
#
# A from-scratch Logistic Regression model that adds
# LENGTH-BASED SAMPLE WEIGHTING to counter the bias:
# "long review = human"


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# the initial skeleton of this code was drawn from google or other sources of a base logistic regression. I wanted to make 
# a modified one but I also didn't see the need to reinvent the wheel so to speak, so i repurposed lines here and there.
# as some chatgpt was used, consider this my note for that.  
def compute_lengths(texts):
    return np.array([len(t.split()) for t in texts], dtype=int)
class LengthWeightedLogisticRegression:
    def __init__(
        self,
        learning_rate=0.1,
        n_epochs=20,
        l2_strength=0.0,
        verbose=False,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.l2_strength = l2_strength
        self.verbose = verbose

        self.weights = None
        self.bias = 0.0 

    def _compute_length_weights(self, lengths, labels):
        buckets = []
        for L in lengths:
            if L < 20:
                buckets.append("short")
            elif L < 50:
                buckets.append("medium")
            else:
                buckets.append("long")

        buckets = np.array(buckets)

        counts = {}
        for b, y in zip(buckets, labels):
            key = (b, int(y))
            counts[key] = counts.get(key, 0) + 1

        raw_weights = {k: 1.0 / v for k, v in counts.items()}

        max_value = max(raw_weights.values())
        for k in raw_weights:
            raw_weights[k] /= max_value

        sample_weights = np.zeros_like(labels, dtype=float)
        for i, (b, y) in enumerate(zip(buckets, labels)):
            sample_weights[i] = raw_weights[(b, int(y))]

        return sample_weights
    #gradient descent.
    def fit(self, X, y, lengths):
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        sample_weights = self._compute_length_weights(lengths, y)

        X_csr = X.tocsr()

        for epoch in range(self.n_epochs):
            scores = np.zeros(n_samples)
            for i in range(n_samples):
                row = X_csr[i]
                dot_product = 0.0
                for idx, value in zip(row.indices, row.data):
                    dot_product += value * self.weights[idx]
                scores[i] = dot_product + self.bias

            probabilities = 1 / (1 + np.exp(-scores))

            # this line below is the main deviation. with this line that takes the sample/review's length into account
            # hopefully it doesn't blindly give long reviews an automatic pass. 
            gradient_factor = (probabilities - y) * sample_weights


            grad_weights = np.zeros(n_features)

            for i in range(n_samples):
                row = X_csr[i]
                gf = gradient_factor[i]
                for idx, value in zip(row.indices, row.data):
                    grad_weights[idx] += value * gf

            grad_weights /= n_samples

            # L2 regularization
            if self.l2_strength > 0:
                grad_weights += self.l2_strength * self.weights

            grad_bias = gradient_factor.mean()


            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias


            if self.verbose:
                eps = 1e-12
                ce = -(y * np.log(probabilities + eps) + (1 - y) * np.log(1 - probabilities + eps))
                loss = np.mean(sample_weights * ce)
                print(f"Epoch {epoch+1}/{self.n_epochs}  Loss={loss:.4f}")

        return self
    def predict_proba(self, X):
        X_csr = X.tocsr()
        n_samples = X_csr.shape[0]
        scores = np.zeros(n_samples)

        # dot product
        for i in range(n_samples):
            row = X_csr[i]
            dot_product = 0.0
            for idx, value in zip(row.indices, row.data):
                dot_product += value * self.weights[idx]
            scores[i] = dot_product + self.bias

        probs_pos = 1 / (1 + np.exp(-scores))
        probs_neg = 1 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

def build_length_weighted_logreg_pipeline():
    return Pipeline([
        ("vectorizer", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            stop_words="english",
        )),
        ("classifier", LengthWeightedLogisticRegression(
            learning_rate=0.1,
            n_epochs=25,
            l2_strength=0.001,
            verbose=True,
        )),
    ])

#it will be noted that I later remembered that @ existed which would've saved a couple lines, bit too late now
