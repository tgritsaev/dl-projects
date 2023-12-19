import numpy as np
from .base import Module
import scipy


class ReLU(Module):
    """
    Applies element-wise ReLU function
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return np.maximum(input, 0)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        mul = np.ones(input.shape)
        mul[input <= 0] = 0
        return grad_output * mul


class Sigmoid(Module):
    """
    Applies element-wise sigmoid function
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :return: array of the same size
        """
        return scipy.special.expit(input)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of an arbitrary size
        :param grad_output: array of the same size
        :return: array of the same size
        """
        expit = scipy.special.expit(input)
        return expit * (1 - expit) * grad_output


class Softmax(Module):
    """
    Applies Softmax operator over the last dimension
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        # credits to https://themaverickmeerkat.com/2019-10-23-Softmax/
        p = scipy.special.softmax(input, axis=1)
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum('ij,ik->ijk', p, p)  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        n = input.shape[1]
        tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        dinput = np.einsum('ijk,ik->ij', dSoftmax, grad_output)
        return dinput


class LogSoftmax(Module):
    """
    Applies LogSoftmax operator over the last dimension
    """

    def compute_output(self, input: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :return: array of the same size
        """
        return scipy.special.log_softmax(input, axis=1)

    def compute_grad_input(self, input: np.array, grad_output: np.array) -> np.array:
        """
        :param input: array of size (batch_size, num_classes)
        :param grad_output: array of the same size
        :return: array of the same size
        """
        p = scipy.special.softmax(input, axis=1)
        jac = np.einsum('ij,ik->ijk', np.ones(p.shape), p)
        return np.einsum("ij,ijk->ik", grad_output, np.eye(p.shape[1]) - jac)
