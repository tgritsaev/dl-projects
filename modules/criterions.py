import numpy as np
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.sum((input - target) ** 2) / (input.shape[0] * input.shape[1])

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2 * (input - target) / (input.shape[0] * input.shape[1])


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """

    def __init__(self):
        super().__init__()
        self.log_softmax = LogSoftmax()

    def compute_output(self, input: np.array, target: np.array) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        log_p = self.log_softmax.compute_output(input)
        c = np.zeros(input.shape)
        c[np.arange(target.shape[0]), target] = 1
        return -np.sum(c * log_p) / input.shape[0]

    def compute_grad_input(self, input: np.array, target: np.array) -> np.array:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        c = np.zeros(input.shape)
        c[np.arange(target.shape[0]), target] = 1
        return -self.log_softmax.compute_grad_input(input, c) / input.shape[0]
