import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        super().__init__()
        self._w = np.random.randn(output_count, input_count) * np.sqrt(2.0 / (input_count + output_count))
        self._b = np.zeros(output_count)

    def get_parameters(self):
        return {'w': self._w, 'b': self._b}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = x @ self._w.T + self._b
        return y, x

    def backward(self, output_grad, cache):
        x = cache
        input_grad = output_grad @ self._w
        w_grad = output_grad.T @ x
        b_grad = np.sum(output_grad, axis=0)
        return input_grad, {'w': w_grad, 'b': b_grad}


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self._alpha = alpha
        self._eps = 1e-8 # Ajouté par l'IA pour éviter les valeur indéfini (x/0)

        self._gamma = np.ones(input_count)
        self._beta = np.zeros(input_count)

        self._global_mean = np.zeros(input_count)
        self._global_variance = np.ones(input_count)

    def get_parameters(self):
        return {'gamma': self._gamma, 'beta': self._beta}

    def get_buffers(self):
        return {'global_mean': self._global_mean, 'global_variance': self._global_variance}

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        return self._forward_evaluation(x)

    def _forward_training(self, x):
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        self._global_mean = (1 - self._alpha) * self._global_mean + self._alpha * mean
        self._global_variance = (1 - self._alpha) * self._global_variance + self._alpha * var

        x_hat = (x - mean) / np.sqrt(var + self._eps)
        y = self._gamma * x_hat + self._beta

        return y, (x, x_hat, mean, var)

    def _forward_evaluation(self, x):
        x_hat = (x - self._global_mean) / np.sqrt(self._global_variance + self._eps)
        y = self._gamma * x_hat + self._beta
        return y, (x, x_hat, self._global_mean, self._global_variance)

    def backward(self, output_grad, cache):
        x, x_hat, mean, var = cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self._eps)

        gamma_grad = np.sum(output_grad * x_hat, axis=0)
        beta_grad = np.sum(output_grad, axis=0)

        dx_hat = output_grad * self._gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * std_inv ** 3, axis=0)
        dmean = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2.0 * (x - mean), axis=0)

        input_grad = dx_hat * std_inv + dvar * 2.0 * (x - mean) / N + dmean / N

        return input_grad, {'gamma': gamma_grad, 'beta': beta_grad}


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1.0 / (1.0 + np.exp(-x))
        return y, y

    def backward(self, output_grad, cache):
        y = cache
        return output_grad * y * (1 - y), {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = np.maximum(0, x)
        return y, x

    def backward(self, output_grad, cache):
        x = cache
        return output_grad * (x > 0), {}