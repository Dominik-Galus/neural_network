from abc import ABC, abstractmethod
from typing import Self

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    @abstractmethod
    def fit(self, learning_data: np.ndarray, observed_data: np.ndarray) -> Self:
        pass

    @abstractmethod
    def _forward(self, x: np.ndarray) -> None:
        pass

    @abstractmethod
    def _backpropagate(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> NDArray[np.int8]:
        pass

    @abstractmethod
    def add_layer(self, neurons: int = 30, activation: str = "relu", inputs: int | None = None) -> None:
        pass
