import numpy as np


def Sigmoid(x):
    return 1/(1+np.exp(-x))


def Softmax(x):
    exp_x = np.exp(x - np.sum(x, axis=1, keepdims=True))
    return exp_x/np.sum(np.exp(x), axis=1, keepdims=True)


def linear(x):
    return x


def ReLU(x):
    mask = x < 0.0
    x[mask] = 0.0
    return x


class DenseLayer:
    def __init__(self, in_features, out_features, act_func='sigmoid'):
        self.weights = np.random.rand(out_features, in_features) * 0.1
        self.biases = np.zeros((1, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.output = None
        if act_func == "sigmoid":
            self.activation_fn = Sigmoid
        elif act_func == "relu":
            self.activation_fn = ReLU
        elif act_func == "softmax":
            self.activation_fn = Softmax
        elif act_func == "linear":
            self.activation_fn = linear
        else:
            raise ValueError(f"Unkown activation function:- {act_func}")

    def __call__(self, inputs):
        weighted_sum = (inputs @ self.weights.T)+self.biases
        activated_sum = self.activation_fn(weighted_sum)
        self.output = activated_sum
        return self.output


class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: DenseLayer):
        assert len(self.layers) == 0 or self.layers[-1].out_features == layer.in_features
        self.layers.append(layer)

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs



if __name__ == '__main__':
    inputs = [
        [2.0, 3.0, 1.0],
        [2.0, 0.5, -1.0],
        [1.0, 2.0, 3.9]
    ]
    inputs = np.array(inputs, ndmin=2)
    print(inputs)
    model = MLP()
    model.add_layer(DenseLayer(3, 6, act_func='relu'))
    model.add_layer(DenseLayer(6, 6, act_func='relu'))
    model.add_layer(DenseLayer(6, 2, act_func='softmax'))
    output = model(inputs)
    print(output, output.shape)
