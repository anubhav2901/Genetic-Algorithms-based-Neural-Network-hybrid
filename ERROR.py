# calculate error
import numpy as np


# mean square error
def mean_square_error(pred, target):
    """
    Determine mean square error.
    f(y_t, y) = sum((y_t-y)**2)/n
        where, y_t = predicted value
                y  = target value
                n = number of values

    :param pred: {array}, shape(n_samples,)
            predicted values.
    :param target: {array}, shape(n_samples,)
            target values.
    :return: mean square error.
    """
    error = pred - target
    square_error = error * error
    return np.mean(square_error)


# mean absolute error
def mean_absolute_error(pred, target):
    """
    Determine mean absolute error.
    f(y_t, y) = sum(abs(y_t-y))/n
        where, y_t = predicted value
                y  = target value
                n = number of values

    :param pred: {array}, shape(n_samples,)
            predicted values.
    :param target: {array}, shape(n_samples,)
            target values.
    :return: mean absolute error.
    """
    abs_error = np.abs(pred - target)
    return np.mean(abs_error)


# mean log cosh error
def mean_log_cosh_error(pred, target):
    """
    Determine mean log cosh error.
    f(y_t, y) = sum(log(cosh(y_t-y)))/n
        where, y_t = predicted value
                y  = target value
                n = number of values

    :param pred: {array}, shape(n_samples,)
            predicted values.
    :param target: {array}, shape(n_samples,)
            target values.
    :return: mean log cosh error.
    """
    error = pred - target
    return np.mean(np.log(np.cosh(error)))


# log loss function
def log_loss(pred, target):
    """
    Determine mean log loss function.
    :param pred: {array}, shape{n_samples,}
            predicted values.
    :param target: {array}, shape{n_samples,}
            target values.
    :return: mean log loss error.
    """
    pred[pred == 0] = 0.00001
    pred[pred == 1] = 0.99999
    return -np.mean((target*np.log(pred) + (1-target)*np.log(1-pred)))


errors = {"mean_square_error": mean_square_error, "mean_absolute_error": mean_absolute_error,
          "mean_log_cosh_error": mean_log_cosh_error, "log_loss": log_loss}
