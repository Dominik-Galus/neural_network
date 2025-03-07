from collections.abc import Callable  # noqa: TC003

import numpy as np

from neural_network.activation_functions.activations import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS


class Layer:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        neurons: int,
        learning_rate: int,
        inputs: int,
        random_state: int = 1,
        activation_function: str = "relu",
        l2: int = 0,
    ) -> None:
        self.activation_function: Callable[[np.ndarray], np.ndarray] | None = ACTIVATION_FUNCTIONS[activation_function]
        self.activation_derivative: Callable[[np.ndarray], np.ndarray] | None = ACTIVATION_DERIVATIVES[
            activation_function
        ]
        if self.activation_function is None or self.activation_derivative is None:
            msg = "There is no such activation function available"
            raise TypeError(msg)

        self.neurons = neurons
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.l2 = l2
        self.weights = np.random.default_rng(random_state).normal(loc=0, scale=0.1, size=(inputs, neurons))
        self.bias = np.ones(shape=neurons)

    def forward(self, x: np.ndarray) -> None:
        pass

    def backward(self, x: np.ndarray) -> None:
        pass
