import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        print("self.weights", self.weights)
        print("self.biases", self.biases)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)

dense1.forward(X)

print(dense1.output[:5])
