import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

class Neuron:
    def __init__(self, in_features: int):
        self.weights = np.random.rand(in_features) * 0.1
        self.weights = self.weights.reshape(1, in_features)
        self.bias = 0.0
    
    def forward(self, inputs_list: list):
        inputs = np.array(inputs_list, ndmin=2)
        
        weighted_sum = np.dot(inputs, self.weights.T)
        outputs = weighted_sum + self.bias
        activated_output = sigmoid(outputs)
        return activated_output, outputs


if __name__ == '__main__':
    neuron = Neuron(in_features=3)
    
    input_list = [1.0, 2.0, 3.0]
    outputs, raw_outputs = neuron.forward(input_list)
    print(outputs, raw_outputs)