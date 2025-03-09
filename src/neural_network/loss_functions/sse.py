import numpy as np
from numpy.typing import NDArray


def sse(y: np.ndarray, output: np.ndarray) -> NDArray[np.float64]:
    error = output - y
    return np.array(0.5 * np.sum(np.square(error)), dtype=np.float64)


def sse_derivative(y: np.ndarray, output: np.ndarray) -> NDArray[np.float64]:
    return np.array(output - y, dtype=np.float64)
