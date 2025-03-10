from copy import copy

import numpy as np
from numpy._typing import NDArray

from neural_network.model import Model


def accuracy_score(observed: NDArray[np.float64], predicted: NDArray[np.int8]) -> float:
    return float(np.sum(observed == predicted) / len(observed))


def cross_validation_score(
    model: Model,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 2,
    ) -> list[float]:
    indices = np.arange(x.shape[0])
    np.random.default_rng().shuffle(indices)

    folds = np.array_split(indices, k)

    accuracy_scores = []
    for i in range(k):
        test_indice = folds[i]
        train_indices = np.hstack([folds[j] for j in range(k) if j != i])

        x_train, x_test = x[train_indices], x[test_indice]
        y_train, y_test = y[train_indices], y[test_indice]

        copied_model = copy(model)
        copied_model.fit(x_train, y_train)

        y_pred = copied_model.predict(x_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    return accuracy_scores
