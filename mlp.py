import numpy as np
class MLP:
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]
    
    def __call__(self, inputs, test=False):
        for layer in self.layers:
            inputs = layer(inputs, test)
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
            layer.batch_weights += -learning_rate * layer.batch_w_grad
            layer.batch_biases += -learning_rate * layer.batch_b_grad

    def save(self, filename):
        data = {}
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_weights"] = layer.weights
            data[f"layer_{i}_batch_weights"] = layer.batch_weights
            data[f"layer_{i}_batch_biases"] = layer.batch_biases
            if layer.running_batch_mean is not None:
                data[f"layer_{i}_running_mean"] = layer.running_batch_mean
                data[f"layer_{i}_running_std"] = layer.running_batch_std
        
        np.savez_compressed(filename, **data)
        print(f"Model saved to {filename}")

    def load(self, filename):
        data = np.load(filename)
        
        for i, layer in enumerate(self.layers):
            w_key = f"layer_{i}_weights"
            batch_w_key = f"layer_{i}_batch_weights"
            batch_b_key = f"layer_{i}_batch_biases"
            mean_key = f"layer_{i}_running_mean"
            std_key = f"layer_{i}_running_std"
            
            if w_key in data and batch_w_key in data and batch_b_key in data and mean_key in data and std_key in data:
                layer.weights = data[w_key]
                layer.batch_weights = data[batch_w_key]
                layer.batch_biases = data[batch_b_key]
                layer.running_batch_mean = data[mean_key]
                layer.running_batch_std = data[std_key]
            else:
                print(f"Warning: Could not find weights for layer {i} in file.")
        
        print(f"Model loaded from {filename}")