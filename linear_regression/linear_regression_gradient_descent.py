import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def loss_function_mse(true_values, predicted_values):
    error = np.sum((true_values-predicted_values) ** 2) / (len(true_values))
    return error

def gradient_descent(bias, lambda1, alpha, X, Y, max_iter):
    mse = []
    for iteration in range(max_iter):

        yPredictions = (lambda1*X)+bias
        error = loss_function_mse(true_values=Y, predicted_values=yPredictions)
        mse.append(error)

        # this is just the same operation
        # np.dot((yPredictions - Y), Y) == np.sum((yPredictions - Y) * Y)

        # calculate derivate w.r.t. lambda
        der_lambda = (1 / len(Y)) * np.sum((Y-yPredictions) * (-2*X))

        # calculate derivate w.r.t. beta
        der_bias = (1 / len(Y)) * np.sum((Y-yPredictions) * -2)

        # update lambda
        lambda1 = lambda1 - (alpha * der_lambda)

        #update bias
        bias = bias - (alpha * der_bias)

    plt.plot(range(max_iter), mse)
    plt.show()

    return bias, lambda1

def linear_regression(X, Y, max_iter=50, learning_rate=0.05):
    # set initial parameters for model
    bias = 0
    lambda1 = 0

    alpha = learning_rate # learning rate
    max_iter= max_iter

    # call gredient decent to calculate intercept(=bias) and slope(lambda1)
    bias, lambda1 = gradient_descent(bias=bias, lambda1=lambda1, alpha=alpha, X=X, Y=Y, max_iter=max_iter)
    print ('Final bias and  lambda1 values are = ', bias, lambda1, " respecively." )

    # plot the data and overlay the linear regression model
    yPredictions = (lambda1*X)+bias
    plt.scatter(X, Y)
    plt.plot(X,yPredictions,'k-')
    plt.show()

def main(X, Y, learning_rate, max_iter):
    # Perform standarization on the feature data
    X = (X - np.mean(X))/np.std(X)
    linear_regression(X, Y, max_iter, learning_rate)

# Read data into a dataframe
df = pd.read_excel('data/data.xlsx')
df = df.dropna()

# Convert Dataframe to a NumPy array
X = df.values
# Store feature and target data in separate arrays
Y = df['Y'].values
X = df['X'].values

main(X, Y, 0.05, 100)
