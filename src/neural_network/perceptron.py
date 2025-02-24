from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 50, random_state: float = 1) -> None:
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.epochs = epochs

    def fit(self, learning_data: np.ndarray, observed_data: np.ndarray) -> Self:
        random_generator = np.random.RandomState(self.random_state)  # type: ignore[arg-type]

        self.weights = random_generator.normal(loc=0.0, scale=0.01, size=1 + learning_data.shape[1])

        self.errors = []
        for _ in range(self.epochs):
            errors = 0
            for xi, expected in zip(learning_data, observed_data, strict=False):
                update = self.learning_rate * (expected - self.predict_value(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def predict_value(self, data: np.ndarray) -> np.ndarray:
        return np.where(self.net_input(data) >= 0.0, 1, -1)

    def net_input(self, data: np.ndarray) -> np.ndarray:
        return np.dot(data, self.weights[1:]) + self.weights[0]  # type: ignore[no-any-return]


def plot_decision_regions(x: np.ndarray, y: np.ndarray, classifier: Perceptron, resolution: float = 0.05) -> None:
    markers = ("x", "o", "s", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    z = classifier.predict_value(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=0.8, c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolors="black",
        )


if __name__ == "__main__":
    iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    y = iris_data.iloc[0:100, 4].to_numpy()
    y = np.where(y == "Iris-setosa", 1, -1)

    x = iris_data.iloc[0:100, [0, 2]].to_numpy()
    ppn = Perceptron(learning_rate=0.1, epochs=10)
    ppn.fit(x, y)
    plot_decision_regions(x, y, classifier=ppn)
    plt.xlabel("Length of sepal [cm]")
    plt.ylabel("Length of petal [cm]")
    plt.legend(loc="upper left")
    plt.show()
