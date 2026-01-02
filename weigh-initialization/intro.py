import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x-np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def linear(x):
    return x


def weight_initialization_Xavier(in_features, out_features):
    stddev = np.sqrt(2.0/(in_features + out_features))
    return np.random.randn(out_features, in_features) * stddev


def weight_initialization_He(in_features, out_features):
    limit = np.sqrt(6.0/(in_features))
    return np.random.uniform(-limit, limit, (out_features, in_features))


def MSE(y_true, y_predict):
    return np.mean((y_true-y_predict)**2)


def CrossEntropyLoss(y_true, y_predict):
    correct = y_predict[y_true==1].flatten()
    return np.mean(-np.log(correct))


class DenseLayer:
    def __init__(self, in_features, out_features, activation_func="sigmoid"):
        self.weights = weight_initialization_Xavier(in_features, out_features)
        self.biases = np.zeros((1, out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.activation_func = sigmoid
        if activation_func == "relu":
            self.activation_func = relu
            self.weights = weight_initialization_He(in_features, out_features)
        elif activation_func == "softmax":
            self.activation_func = softmax
        elif activation_func == "linear":
            self.activation_func = linear
        else:
            print(f"using default:- {activation_func}")

    def __call__(self, inputs):
        weighted_sum = (inputs @ self.weights.T) + self.biases
        activation_result = self.activation_func(weighted_sum)
        return activation_result


class MLP:
    def __init__(self):
        self.layers = []
    
    def add_layers(self, layer):
        assert len(self.layers)==0 or self.layers[-1].out_features == layer.in_features
        self.layers.append(layer)
    
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


if __name__ == '__main__':
    ## LLM task
    inputs = [
        [1.0, 2.0, -1.3, 2.3],
        [2.0, 3.9, 1.0, -0.34],
        [3.0, -0.7, 1.2, 0.2]
    ]
    test_outputs = [[0, 1], [1, 0], [1, 0]]
    inputs = np.array(inputs, ndmin=2)
    test_outputs = np.array(test_outputs, ndmin=2)
    model = MLP()
    model.add_layers(DenseLayer(4, 10, "relu"))
    model.add_layers(DenseLayer(10, 10, "relu"))
    model.add_layers(DenseLayer(10, 2, "softmax"))
    outputs = model(inputs)
    print(outputs)
    print(f"Loss:- {CrossEntropyLoss(test_outputs, outputs)}")
    
    ## Regression:-
    inputs = [
        [1.0, 2.0, -1.3, 2.3],
        [2.0, 3.9, 1.0, -0.34],
        [3.0, -0.7, 1.2, 0.2]
    ]
    test_outputs = [1.0, 2.0, 1.5]
    inputs = np.array(inputs, ndmin=2)
    test_outputs = np.array(test_outputs, ndmin=2).reshape(-1, 1)
    model = MLP()
    model.add_layers(DenseLayer(4, 10, "relu"))
    model.add_layers(DenseLayer(10, 10, "relu"))
    model.add_layers(DenseLayer(10, 1, "linear"))
    outputs = model(inputs)
    print(outputs)
    print(f"Loss:- {MSE(test_outputs, outputs)}")