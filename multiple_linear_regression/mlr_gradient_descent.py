import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

def hypothesis(X, coefficients, bias):
    """ This function will take in all the feature data X
    as well as the current coefficient and bias values
    It should multiply all the feature value by their associated
    coefficient and add the bias. It should then return the predicted
    y values.
    """
    predicted = np.dot(coefficients, X.transpose()) + bias
    return predicted

def calculate_r_squared(bias, coefficients, X, Y):
    # R squared calculation from scratch
    predictedY = hypothesis(X, coefficients, bias)
    avg_y = np.average(Y)
    total_sum_sq = np.sum((avg_y - Y)**2)
    sum_sq_res = np.sum((predictedY - Y)**2)
    r2 = 1.0-(sum_sq_res/total_sum_sq)
    return r2


def gradient_descent(bias, coefficients, alpha, X, Y, max_iter):
    # an array is used to store change in cost function for each iteration of GD
    errorValues = []
    length = len(X)
    for num in range(0, max_iter):
        # calculate predicted y values for current coefficient and bias values
        # calculate and update bias using gradient descent rule
        # Update each coefficient value in turn using gradient descent rule
        predicted = hypothesis(X, coefficients, bias)
        error = np.subtract(predicted, Y)
        # derivatives
        der_bias = ((1) / (len(Y))) * np.sum(error)
        der_lambda = ((1) / (len(Y))) * np.dot(error, X)
        # coefficients and bias
        coefficients = coefficients - (alpha * der_lambda)
        bias = bias - (alpha * der_bias)
        # calculate cost
        cost = (1.0 / ((length) * 2)) * (np.sum((predicted - Y) ** 2))
        errorValues.append(cost)

    # calculate R squared value for current coefficient and bias values
    r_squared = calculate_r_squared(bias, coefficients, X, Y)
    print ("Final R2 value is ", r_squared)
    # plot the cost for each iteration of gradient descent
    plt.plot(errorValues)
    plt.legend()
    plt.suptitle('MLR model | alpha = {} | max_iter = {} | Boston dataset'.format(alpha, max_iter))
    plt.title('Cost Function = (1 / (2 * length)) * (np.sum((predicted - Y) ** 2))', fontsize=10)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    return bias, coefficients

def multiple_linear_regression(X, Y):
    # set the number of coefficients equal to the number of features
    coefficients = np.random.uniform(low=0, high=0.5, size=X.shape[1])
    bias = 5.0
    alpha = 0.1 # learning rate
    max_iter = 100
    # call gredient decent, and get intercept(=bias) and coefficents
    bias, coefficients = gradient_descent(bias, coefficients, alpha, X, Y, max_iter)
    return bias, coefficients

def main(dataset):
    df = pd.read_csv(dataset)
    df = df.dropna()
    print(df.shape)
    data = df.values
    # Seperate teh features from the target feature
    Y = data[:, -1]
    X = data[:, :-1]
    # Standardize each of the features in the dataset.
    for num in range(len(X[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        X[:, num] = feature
    # run regression function
    bias, coefficients = multiple_linear_regression(X, Y)
    return bias, coefficients

def calculate_test_accuracy(bias, coefficients, dataset):
    df = pd.read_csv(dataset)
    df = df.dropna()
    data = df.values
    # Seperate the features from the target feature
    testY = data[:, -1]
    testX = data[:, :-1]
        # Standardize each of the features in the dataset.
    for num in range(len(testX[0])):
        feature = data[:, num]
        feature = (feature - np.mean(feature))/np.std(feature)
        testX[:, num] = feature
    r_squared = calculate_r_squared(bias, coefficients, testX, testY)
    print ("Test R2 value is ", r_squared)

if __name__ == "__main__":
    bias, coefficients = main(dataset='data/trainingData.csv')
    calculate_test_accuracy(bias, coefficients, 'data/testData.csv')
