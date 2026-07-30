import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        res = -np.mean(y_true*np.log(y_pred+epsilon) + (1-y_true)*np.log(1-y_pred+epsilon))
        return round(res, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        inner = np.zeros(len(y_true))
        for i in range(len(y_true)):
            inner[i] = np.sum(y_true[i]*np.log(y_pred[i] + epsilon))
        outer = -np.mean(inner)
        return round(outer, 4)
