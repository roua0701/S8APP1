import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        N = x.shape[0]
        probs = softmax(x)

        loss = -np.mean(np.log(probs[np.arange(N), target]))

        one_hot = np.zeros_like(x)
        one_hot[np.arange(N), target] = 1
        input_grad = (probs - one_hot) / N

        return loss, input_grad


def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True) # Sécurité pour ne pas avoir de trop grandes valeurs
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        diff = x - target
        loss = np.mean(diff ** 2)
        input_grad = 2 * diff / diff.size

        return loss, input_grad