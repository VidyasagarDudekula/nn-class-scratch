import numpy as np
class MLP:
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]
    
    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
    
    def __repr__(self) -> str:
        s = ""
        for i, layer in enumerate(self.layers):
            s += f"layer{i+1}: weights {layer.weights.shape},\n"
        return s
    
    def backward(self, loss_value):
        for layer in self.layers[::-1]:
            loss_value = layer.backward(loss_value)
        return loss_value
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def step(self, learning_rate=0.01):
        for layer in self.layers[::-1]:
            layer.weights += -learning_rate * layer.w_grad
            layer.biases += -learning_rate * layer.b_grad

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_weights"] = layer.weights
            data[f"layer_{i}_biases"] = layer.biases
        
        np.savez_compressed(filename, **data)
        print(f"Model saved to {filename}")

    def load(self, filename):
        data = np.load(filename)
        
        for i, layer in enumerate(self.layers):
            w_key = f"layer_{i}_weights"
            b_key = f"layer_{i}_biases"
            
            if w_key in data and b_key in data:
                layer.weights = data[w_key]
                layer.biases = data[b_key]
            else:
                print(f"Warning: Could not find weights for layer {i} in file.")
        
        print(f"Model loaded from {filename}")