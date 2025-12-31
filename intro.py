import numpy as np


class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.rand(n_inputs) * 0.1
        self.weights = self.weights.reshape(1, n_inputs)
        self.bias = 0.0
    
    def forward(self, inputs):
        inputs = np.array(inputs, ndmin=2)
        print(inputs.shape)
        weighted_sum = np.dot(inputs, self.weights.T)
        output = weighted_sum + self.bias
        return output


if __name__ == '__main__':
    n_input = 3
    neuron = Neuron(n_inputs=n_input)
    input_array = [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 1.0],
        [2.0, 3.0, 4.0],
        [1.0, 3.0, 4.0]
    ]
    output = neuron.forward(input_array)
    print(neuron.weights.shape)
    print(output)