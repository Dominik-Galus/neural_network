from typing import TYPE_CHECKING

import numpy as np

from neural_network.loss_functions.loss_functions import LOSS_DERIVATIVES, LOSS_FUNCTIONS

if TYPE_CHECKING:
    from neural_network.layer import Layer


class Perceptron:
    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 50,
        l2: float = 0.01,
        loss_function: str = "sse",
        batch_size: int = 8,
    ) -> None:
        loss_function_callable = LOSS_FUNCTIONS.get(loss_function)
        if loss_function_callable is None:
            msg = "There is no such loss function."
            raise TypeError(msg)

        loss_derivative_callable = LOSS_DERIVATIVES.get(loss_function)
        if loss_derivative_callable is None:
            msg = "There is no such loss function."
            raise TypeError(msg)

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.loss_function = loss_function_callable
        self.loss_derivative = loss_derivative_callable
        self.batch_size = batch_size
        self.layers: list[Layer] = []

    def fit(self, learning_data: np.ndarray, observed_data: np.ndarray) -> None:
        pass

    def _forward(self, x: np.ndarray) -> None:
        self.a: list[np.ndarray] = []
        self.z: list[np.ndarray] = []

        a, z = self.layers[0].forward(x)
        self.a.append(a)
        self.z.append(z)

        for idx in range(1, len(self.layers)):
            a, z = self.layers[idx].forward(a)
            self.a.append(a)
            self.z.append(z)

    def _backpropagate(self, x: np.ndarray, y: np.ndarray) -> None:

        delta = self.loss_derivative(y, self.a[-1]) * self.layers[-1].activation_derivative(self.z[-1])

        for idx in reversed(range(1, len(self.layers))):
            delta = self.layers[idx].backward(self.a[idx - 1], delta, self.z[idx - 1])
        self.layers[0].backward(x, delta)

    def predict_value(self, data: np.ndarray) -> None:
        pass

    def net_input(self, data: np.ndarray) -> None:
        pass
