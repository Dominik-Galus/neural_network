import numpy as np
from numpy.typing import NDArray


def sigmoid_activation(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return sigmoid_activation(x) * (1. - sigmoid_activation(x))
