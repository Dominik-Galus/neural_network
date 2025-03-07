from collections.abc import Callable

from neural_network.activation_functions.relu import relu_activation, relu_derivative
from neural_network.activation_functions.sigmoid import sigmoid_activation, sigmoid_derivative

ACTIVATION_FUNCTIONS: dict[str, Callable] = {
    "relu": relu_activation,
    "sigmoid": sigmoid_activation,
}

ACTIVATION_DERIVATIVES: dict[str, Callable] = {
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
}
