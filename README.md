# Neural Network Implementation

This repository contains an implementation of a simple neural network using Python and NumPy. The neural network is built with fully connected layers and supports multiple activation and loss functions. The implementation is designed to train on the MNIST dataset.

## Features
- Fully connected neural network with customizable layers
- Supports different activation functions (e.g., ReLU, Sigmoid)
- Implements L2 regularization
- Backpropagation and weight updates with gradient descent
- Batch training with mini-batches
- Supports multiple loss functions (e.g., SSE, Cross-Entropy)
- Validation and accuracy tracking
- Includes visualization utilities for cost function and predictions

## Requirements
To run this project, install the following dependencies:
```bash
pip install numpy tqdm matplotlib
```
or
```bash
pip install .
```

### Adding Layers
You can add layers to the neural network before training:
```python
model = Perceptron(epochs=25, loss_function="cross_entropy")
model.add_layer(50, "relu", 784)  # Hidden layer with 50 neurons
model.add_layer(10, "sigmoid")  # Output layer with 10 neurons (for MNIST digits)
```
### Model Training
```python
model.fit(x_train, y_train, validation_split=0.1)
```
validation_split parameter is not required.

### Making Predictions
```python
y_pred = model.predict(x_test)
```
### Visualizing Results
![cross_val](https://github.com/user-attachments/assets/acbc6ab8-f400-42f3-a554-02a57a30bc34)
![cost_plot](https://github.com/user-attachments/assets/330dc7b1-7789-4ddc-989d-61d60f64cb84)
![examples](https://github.com/user-attachments/assets/753e2416-9518-474d-87c7-881595eb8a9f)

