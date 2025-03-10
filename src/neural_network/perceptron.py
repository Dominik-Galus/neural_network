from typing import Self

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from neural_network.datasets.load_mnist import load_mnist
from neural_network.layer import Layer
from neural_network.loss_functions.loss_functions import LOSS_DERIVATIVES, LOSS_FUNCTIONS
from neural_network.model import Model
from neural_network.preprocessing.one_hot import one_hot


class Perceptron(Model):
    def __init__(
        self,
        learning_rate: float = 0.001,
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

    def fit(self, learning_data: np.ndarray, observed_data: np.ndarray) -> Self:
        observed_data_enc = one_hot(observed_data, np.unique(observed_data).shape[0])
        learning_data_flatten = learning_data.reshape(
            learning_data.shape[0],
            learning_data.shape[1] * learning_data.shape[2],
        )
        self.cost_: list[np.ndarray] = []

        for _ in tqdm(range(self.epochs), ascii=True, desc="Epochs"):
            indices = np.arange(learning_data.shape[0])
            np.random.default_rng().shuffle(indices)

            for k in range(0, indices.shape[0] - self.batch_size + 1, self.batch_size):
                batched = indices[k:k + self.batch_size]

                self._forward(learning_data_flatten[batched])
                self._backpropagate(learning_data_flatten[batched], observed_data_enc[batched])

            self._forward(learning_data_flatten)
            self.cost_.append(self.loss_function(self.a[-1], observed_data_enc))

        return self

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

    def predict(self, data: np.ndarray) -> NDArray[np.int8]:
        data_flatten = data.reshape(
            data.shape[0],
            data.shape[1] * data.shape[2],
        )
        self._forward(data_flatten)
        return np.array(np.argmax(self.z[-1], axis=1), dtype=np.int8)

    def add_layer(self, neurons: int = 30, activation: str = "relu", inputs: int | None = None) -> None:
        if not self.layers and not inputs:
            msg = "There is no input layer in neuraln network"
            raise ValueError(msg)
        if not inputs:
            self.layers.append(Layer(
                neurons=neurons,
                inputs=self.layers[-1].neurons,
                learning_rate=self.learning_rate,
                activation_function=activation,
                l2=self.l2))
        else:
            self.layers.append(Layer(
                neurons=neurons,
                inputs=inputs,
                learning_rate=self.learning_rate,
                activation_function=activation,
                l2=self.l2))


if __name__ == "__main__":
    x_train, y_train = load_mnist()
    x_test, y_test = load_mnist(kind="test")

    model = Perceptron(epochs=25, loss_function="cross_entropy")
    model.add_layer(50, "relu", 784)
    model.add_layer(np.unique(y_train).shape[0], "sigmoid")
    model.fit(x_train[4000:], y_train[4000:])

    x_test_1 = x_test.reshape(
        x_test.shape[0],
        x_test.shape[1] * x_test.shape[2],
    )

    y_pred = model.predict(x_test)

    import matplotlib.pyplot as plt
    plt.plot(range(model.epochs), model.cost_)
    plt.xlabel("Epoki")
    plt.ylabel("Koszt")
    plt.show()

    fig, ax = plt.subplots(10, 4, figsize=(10, 15))
    for i in range(40):
        plt.subplot(10, 4, i + 1)
        plt.imshow(x_test[i])
        plt.title(f"Predicted Digit: {y_pred[i]}")
    fig.tight_layout()
    plt.show()
