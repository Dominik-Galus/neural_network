import numpy as np
from numpy.typing import NDArray


def relu_activation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(x, 0)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)
