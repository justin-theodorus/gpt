import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        total = np.sum(np.exp(z - z.max()))
        return np.round(np.exp(z - z.max()) / total,4)
