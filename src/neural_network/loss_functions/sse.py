import numpy as np
from numpy.typing import NDArray


def sse(x: np.ndarray, y: np.ndarray) -> NDArray[np.float64]:
    error = y - x
    return np.array(0.5 * np.sum(np.square(error)), dtype=np.float64)


def sse_derivative(x: np.ndarray, y: np.ndarray) -> NDArray[np.float64]:
    return np.array(y - x, dtype=np.float64)
