from collections.abc import Callable

import numpy as np

from neural_network.loss_functions.cross_entropy import cross_entropy, cross_entropy_derivative
from neural_network.loss_functions.sse import sse, sse_derivative

LOSS_FUNCTIONS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "sse": sse,
    "cross_entropy": cross_entropy,
}

LOSS_DERIVATIVES: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "sse": sse_derivative,
    "cross_entropy": cross_entropy_derivative,
}
