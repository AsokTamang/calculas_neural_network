# Neural Network Mini Project: Linear and Multiple Regression

## Overview
This project demonstrates how to implement a **simple neural network** to perform regression tasks using Python and NumPy. The focus is on:

1. Building a **simple linear regression model** using a neural network to predict sales given a TV marketing budget.
2. Extending the model to a **multiple linear regression scenario** to predict house prices based on size (`GrLivArea`) and quality (`OverallQual`).
3. Implementing **gradient descent** manually to train the network.
4. Normalizing and denormalizing data for accurate predictions.

---

## Project Structure
.
├── data/
│ └── housing_data.csv # Sample dataset for house prices
├── utils.py # Utility functions (forward/backward propagation, normalization)
├── README.md # Project documentation


---

## Implementation Details

### 1. Neural Network Architecture

- **Input layer:** Number of features (1 for TV sales, 2 for house prices)
- **Output layer:** Single neuron (predicted sales or price)
- **No hidden layers** for simplicity, corresponding to linear regression.
- **Weights and biases** initialized randomly.
- **Forward propagation:** Computes predicted outputs.
- **Cost function:** Mean squared error (MSE)
- **Backward propagation:** Computes gradients with respect to weights and biases.
- **Gradient descent:** Updates parameters iteratively to minimize cost.

---

### 2. Key Functions

#### `layer_sizes(X, Y)`
Determines the number of input and output neurons based on the dataset.

#### `initialize_parameters(n_x, n_y)`
Initializes weights and biases for the network.

#### `forward_propagation(X, parameters)`
Performs the forward pass to calculate predictions.

#### `cost_function(Y_predicted, Y)`
Calculates the mean squared error between predictions and actual values.

#### `backward_propagation(Y_predicted, Y, X)`
Calculates gradients for weights and biases.

#### `update_parameters(parameters, gradients, learning_rate)`
Updates weights and biases using gradient descent.

#### `neural_network_model(X, Y, iterations, learning_rate)`
Trains the neural network over a specified number of iterations.

#### `predict_value(X, Y, X_predict, parameters)`
Predicts new outputs, handling normalization and denormalization.

---




