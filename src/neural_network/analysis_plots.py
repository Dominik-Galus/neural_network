import matplotlib.pyplot as plt
import numpy as np


def predict_examples(sample_data: np.ndarray, predicted: np.ndarray, start: int = 1, end: int = 40) -> None:
    fig, _ = plt.subplots(10, 4, figsize=(10, 15))
    for i in range(end):
        plt.subplot(10, 4, i + start)
        plt.imshow(sample_data[i])
        plt.title(f"Predicted Digit: {predicted[i]}")
    fig.tight_layout()


def cost_plot(epochs: int, cost: list[np.ndarray]) -> None:
    plt.plot(range(epochs), cost)
    plt.xlabel("Epochs")
    plt.ylabel("Cost every epoch")
