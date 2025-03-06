import pathlib
import struct

import numpy as np


def load_mnist(kind: str = "train") -> tuple[np.ndarray, np.ndarray]:
    labels_path = pathlib.Path(f"MNIST/{kind}-labels-idx1-ubyte")
    images_path = pathlib.Path(f"MNIST/{kind}-images-idx3-ubyte")

    with pathlib.Path.open(labels_path, "rb") as lbpath:
        struct.unpack(">II", lbpath.read(8))

        labels = np.fromfile(lbpath, dtype=np.uint8)

    with pathlib.Path.open(images_path, "rb") as impath:
        _, num, rows, cols = struct.unpack(">IIII", impath.read(16))

        images = np.fromfile(impath, dtype=np.uint8).reshape(num, rows, cols)

        images = ((images / 255.) - .5) * 2  # type: ignore[assignment]

    return images, labels


if __name__ == "__main__":
    x, y = load_mnist()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = x[y == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap="Greys")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()

    example_digit: int = 7
    for i in range(10):
        img = x[y == example_digit][i + 30]
        ax[i].imshow(img, cmap="Greys")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
