import numpy as np
from scipy import sparse

class CustomLinearSVM:
    """Linear SVM with hinge loss"""

    def __init__(self, lr = 0.005, C = 0.1, max_iter = 5000):
        self.lr = lr
        self.C = C
        self.max_iter = max_iter
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        # Ensure X is in CSR (fast row access)
        # (used AI for this fix)
        if not sparse.isspmatrix_csr(X):
            X = sparse.csr_matrix(X)

        # convert 0,1 to -1, 1
        y = np.where(y == 1, 1.0, -1.0)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.max_iter):
            # margin = y_i * (w * x_i + b)
            margins = y * (X.dot(self.w) + self.b)

            misclassified = margins < 1
            if not np.any(misclassified):
                # regularization
                dw = self.w
                db = 0.0
            else:
                X_mis = X[misclassified]
                y_mis = y[misclassified]

                # sum(y_i * x_i) using matrix ops
                grad_vec = X_mis.multiply(y_mis[:, None]).sum(axis=0)

                # convert to 1D array
                grad_vec = np.asarray(grad_vec).ravel()

                dw = self.w - self.C * grad_vec
                db = -self.C * np.sum(y_mis)

            # update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

        return self

    def decision_function(self, X):
        if sparse.issparse(X):
            return X.dot(self.w) + self.b
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)
