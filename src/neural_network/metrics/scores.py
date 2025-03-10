import numpy as np
from numpy._typing import NDArray

from neural_network.perceptron import Perceptron


def accuracy_score(observed: NDArray[np.int8], predicted: NDArray[np.int8]) -> float:
    return float(np.sum(observed == predicted) / len(observed))


def cross_validation_score(model: Perceptron, x: np.ndarray, y: np.ndarray, cv: int = 5) -> None:
    pass
