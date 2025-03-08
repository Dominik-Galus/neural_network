from collections.abc import Callable  # noqa: TC003

import numpy as np

from neural_network.activation_functions.activations import ACTIVATION_DERIVATIVES, ACTIVATION_FUNCTIONS


class Layer:
    def __init__(  # noqa: PLR0913, PLR0917
        self,
        neurons: int,
        learning_rate: float,
        inputs: int,
        random_state: int = 1,
        activation_function: str = "relu",
        l2: float = 0,
    ) -> None:
        self.activation_function: Callable[[np.ndarray], np.ndarray] | None = ACTIVATION_FUNCTIONS[activation_function]
        self.activation_derivative: Callable[[np.ndarray], np.ndarray] | None = ACTIVATION_DERIVATIVES[
            activation_function
        ]
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.l2 = l2
        self.weights = np.random.default_rng(random_state).normal(loc=0, scale=0.1, size=(inputs, neurons))
        self.bias = np.zeros(shape=neurons)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.activation_function is None:
            msg = "There is no such activation function available"
            raise TypeError(msg)

        z = np.dot(x, self.weights) + self.bias
        a = self.activation_function(z)

        return a, z

    def backward(self, a: np.ndarray, z: np.ndarray, loss_derivative: np.ndarray) -> None:
        if self.activation_derivative is None:
            msg = "There is no such activation function available"
            raise TypeError(msg)

        delta = self.activation_derivative(z) * loss_derivative
        update_weight = np.dot(a.T, delta)
        update_bias = np.sum(delta)

        self.weights -= (update_weight + self.l2 * self.weights) * self.learning_rate
        self.bias -= update_bias * self.learning_rate
