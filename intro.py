import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class DenseLayer:
    def __init__(self, in_features, out_features):
        self.weights = np.random.rand(out_features, in_features) * 0.1
        self.biases = np.zeros((1, out_features))
    
    def __call__(self, inputs):
        weighted_sum = (inputs @ self.weights.T)+self.biases
        activated_sum = sigmoid(weighted_sum)
        return activated_sum


if __name__ == '__main__':
    inputs = [
        [2.0, 3.0, 1.0],
        [2.0, 0.5, -1.0],
        [1.0, 2.0, 3.9]
    ]
    in_f = 3
    out_f = 1
    inputs = np.array(inputs, ndmin=2)
    layer = DenseLayer(in_f, out_f)
    print(layer(inputs))
