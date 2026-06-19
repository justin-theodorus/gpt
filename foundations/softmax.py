import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        transformed = np.exp(z - z.max())
        return np.round(transformed / np.sum(transformed), 4)
