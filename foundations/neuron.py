import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        def activation_fun(z: NDArray[np.float64], activation: str)-> NDArray[np.float64]:
            if activation == 'sigmoid':
                return self.sigmoid(z)
            if activation == 'relu':
                return self.relu(z)

        preactivation = np.dot(x,w) + b
        activ = activation_fun(preactivation, activation)
        return np.round(activ, 5)

    def relu(self, z):
            return np.maximum(0,z)

    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))

        