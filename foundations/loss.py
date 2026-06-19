import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        n = y_true.shape[0]
        total = -np.sum(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)) / n
        return np.round(total, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        n = y_true.shape[0]
        total = -np.sum(y_true * np.log(y_pred + eps)) / n
        return np.round(total, 4)
