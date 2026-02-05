import numpy as np
import matplotlib.pyplot as plt
import pandas as pd





def layer_sizes(X, Y):
    n_x = X.shape[0]  # rows of the input layer or the number of features in the input dataset
    n_y = Y.shape[0]  # row of the output

    return (n_x, n_y)


def initialize_parameters(n_x,n_y):
    w = np.random.randn(n_y,n_x)*0.1  #generating a random value of size of  output neuron's row as row and the input neuron's row as the column
    b = np.zeros((n_y,1))
    parameters = {
        'w':w,
        'b':b
    }
    return parameters


def forward_propagation(X,parameters):  #this function gives us the predicted values
    weight = parameters['w']
    bias = parameters['b']
    output = (weight @ X) + bias  #matrix multiplication between weight and the input
    return output


def cost_function(Y_predicted,Y):
    m=Y_predicted.shape[1]  #the total number of predicted data points of the passed number of samples 'm'
    return np.sum((Y_predicted - Y)**2) / (2*m)


def backward_propagation(Y_predicted,Y,X):  #here in backward_propagation function, we calculate the partial derivative of loss function with respect to weight of innput and the bias
    dLY = Y_predicted - Y  #partial derivative of loss function with respect to predicted data
    m = Y_predicted.shape[1]
    #the summation of the partial derivatives of loss function with respect to weight and the bias
    dLW = np.dot(dLY,X.T) / m   #partial derivative of loss function with respect to input #transposing the input X  for the matrix multiplication
    dLB = np.sum(dLY,axis = 1) / m    #partial derivative of loss function with respect to the bias
    grads = {
        'dw':dLW,
        'db':dLB
    }
    return grads


def update_parameters(parameters,gradients,learning_rate=0.1):
    weight = parameters['w']
    bias = parameters['b']
    dw = gradients['dw']
    db = gradients['db']
    weight = weight - (learning_rate * dw)
    bias = bias - (learning_rate * db)
    parameters = {
        'w':weight,
        'b':bias
    }
    return parameters


def neural_network_model(X, Y, iterations, learning_rate=0.1, print_cost=False):
    (n_x, n_y) = layer_sizes(X, Y)  # size of input and output neuron respectively
    parameters = initialize_parameters(n_x, n_y)  # the initial parameters
    for i in range(iterations):
        Y_predicted = forward_propagation(X, parameters)
        cost = cost_function(Y_predicted, Y)
        grads = backward_propagation(Y_predicted, Y, X)
        parameters = update_parameters(parameters, grads, learning_rate=0.1)
        if print_cost:
            print(f'Cost function after iteration {i} is', cost)

    return parameters


def predict_value(X, Y, X_predict, parameters):
    X_mean = X.mean().values.reshape(-1,1)
    X_std = X.std().values.reshape(-1,1)
    X_predict_norm = (X_predict - X_mean) / X_std   # broadcasting works for (n_features, n_samples)
    output = forward_propagation(X_predict_norm, parameters)
    return (output * Y.std()) + Y.mean()
