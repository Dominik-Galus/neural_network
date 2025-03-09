import numpy as np
from numpy.typing import NDArray


def cross_entropy(y: np.ndarray, output: np.ndarray, epsilon: float = 1e-12) -> NDArray[np.float64]:
    output = np.clip(output, epsilon, 1. - epsilon)
    return np.array(np.sum(-y * np.log(output) - (1. - y) * np.log(1. - output)), dtype=np.float64)


def cross_entropy_derivative(y: np.ndarray, output: np.ndarray, epsilon: float = 1e-12) -> NDArray[np.float64]:
    output = np.clip(output, epsilon, 1. - epsilon)
    return np.array((output - y) / ((1. - output) * output), dtype=np.float64)
