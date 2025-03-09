import numpy as np
from numpy._typing import NDArray

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
        activation_function_callable = ACTIVATION_FUNCTIONS.get(activation_function)
        if activation_function_callable is None:
            msg = "There is no such activation function available"
            raise TypeError(msg)

        activation_derivative_callable = ACTIVATION_DERIVATIVES.get(activation_function)
        if activation_derivative_callable is None:
            msg = "There is no such activation function available"
            raise TypeError(msg)

        self.activation_function = activation_function_callable
        self.activation_derivative = activation_derivative_callable
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.l2 = l2
        self.weights = np.random.default_rng(random_state).normal(loc=0, scale=0.1, size=(inputs, neurons))
        self.bias = np.zeros(shape=neurons)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.dot(x, self.weights) + self.bias
        a = self.activation_function(z)

        return a, z

    def backward(self, a: np.ndarray, delta: np.ndarray, z: np.ndarray | None = None) -> NDArray[np.float64] | None:
        update_weight = np.dot(a.T, delta)
        update_bias = np.sum(delta, axis=0)

        self.weights -= (update_weight + self.l2 * self.weights) * self.learning_rate
        self.bias -= update_bias * self.learning_rate

        if z is not None:
            return np.array(self.activation_derivative(z) * np.dot(delta, self.weights.T), dtype=np.float64)
        return None
