import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


class DenseLayer:
    def __init__(self, in_features: int, out_features: int):
        self.weights = np.random.rand(out_features, in_features) * 0.1
        self.biases = np.zeros((1, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.outputs = None

    def forward(self, inputs):
        weighted_sum = (inputs @ self.weights.T) + self.biases
        self.outputs = sigmoid(weighted_sum)
        return self.outputs


if __name__ == '__main__':
    inputs = [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 1.0, 1.0], [2.0, 1.0, 1.0, 0.91]]
    inputs = np.array(inputs, ndmin=2)
    in_features = 4
    out_features = 3
    layer = DenseLayer(in_features, out_features)
    outputs = layer.forward(inputs)
    print(outputs)
