import numpy as np


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
        self.activation_function = activation_function
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.inputs = inputs
        self.l2 = l2
        self.weights = np.random.default_rng(random_state).normal(loc=0, scale=0.1, size=(inputs, neurons))

    def forward(self, x: np.ndarray) -> None:
        pass

    def backward(self, x: np.ndarray) -> None:
        pass
