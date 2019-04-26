# for activation functions of Neural Networks
import numpy as np


# activation function
def linear(x):
    """
    Performs linear function on the input elements.
    Linear function: f(x) = x

    :param x: {array}, shape {n_samples,}
            input to the function.
    :return: {array}, shape {n_samples,}
            output of linear function.
    """
    return x


def logistic(x):
    """
    Performs logistic function on the input elements.
    Logistic function: f(x) = 1/(1+exp(-x))

    :param x: {array}, shape {n_samples,}
            input to the function.
    :return: {array}, shape {n_samples,}
            output of logistic function.
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    """
    Performs Rectified Linear Unit function on the input elements.
    Rectified Linear Unit: f(x) = max(0,x)

    :param x: {array}, shape {n_samples,}
            input to the function.
    :return: {array}, shape {n_samples,}
            output of relu function.
    """
    return np.maximum(x, 0)


def tanh(x):
    """
    Performs Bipolar logistic function on the input elements.
    Bipolar logistic function: f(x) = tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
                                    = 2*(1/(1+exp(-2*x)) - 1
    :param x: {array}, shape {n_samples,}
            input to the function.
    :return: {array}, shape {n_samples,}
            output of tanh function.
    """
    return 2 * logistic(x) - np.ones(x.shape)


def rbf(x):
    """
    Performs Radial Basis function on the input elements.
    Radial Basis function: f(x) = exp(-(x - mean_x)**2/(2*var_x)) / sqrt(2*pi*var_x)

    :param x: {array}, shape {n_samples,}
            input to the function.
    :return: {array}, shape {n_samples,}
            output of rbf function.
    """
    mean_x = np.mean(x)
    var_x = np.var(x)
    z = (x - mean_x)
    return np.exp(-z**2 / (2 * var_x)) / np.sqrt(2 * np.pi * var_x)


activations = {"logistic": logistic, "relu": relu,
               "tanh": tanh, "rbf": rbf, "linear": linear}