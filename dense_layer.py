import numpy as np
import functions as F


class DenseLayer:
    def __init__(self, in_features, out_features, activation_func ="linear"):
        self.weights = F.RandomWeights(in_features, out_features)
        self.biases = np.zeros((1, out_features))
        
        self.activation_function = F.linear
        
        self.in_features = in_features
        self.out_features = out_features
        self.derivative = None
        
        self.outputs = None
        self.z = None
        self.w_grad = None
        self.b_grad = None
        
        if activation_func == "sigmoid":
            self.activation_function = F.sigmoid
            self.weights = F.XavierWeights(in_features, out_features)
            self.derivative = F.sigmoid_derivative
        elif activation_func == "softmax":
            self.activation_function = F.softmax
            self.weights = F.XavierWeights(in_features, out_features)
            self.derivative = F.softmax_derivative
        elif activation_func == "linear":
            self.activation_function = F.linear
            self.weights = F.XavierWeights(in_features, out_features)
            self.derivative = F.linear_derivative
        elif activation_func == "relu":
            self.activation_function = F.ReLU
            self.weights = F.HeWeights(in_features, out_features)
            self.derivative = F.ReLU_derivative


    def __call__(self, inputs):
        self.inputs = inputs
        self.z = inputs @ self.weights.T + self.biases
        self.outputs = self.activation_function(self.z)
        return self.outputs
    
    def zero_grad(self):
        self.w_grad = 0.0
        self.b_grad = 0.0
        self.inputs = None
        self.outputs = None
    
    # to-do write back propogation your own.
    def backward(self, derivative_prev_layer):
        # original forward :-
        """
        z = inputs @ self.weights.T + self.biases --> z = xW + b
        y = activation(z)
        
        Lets go from bottom to top, as that is the right order to derivate backwards.
        first dl/dy -> loss with repspective to y --> derivate of activation function * prev_derivative (or grad) -> input to this will be self.outputs as 
        that is the output from the activation in forward pass.
        dl/dy = derivative_prev_layer * self.derivative(self.outputs)
        
        second:- z = xW + b
        dl/dz = dl/dy * dy/dz
        z is made up of W, b, x
        dl/dw -> dl/dz * dz/dw
        dz/dw --> x  --> w_grad
        dz/db --> 1 --> b_grad
        dz/dx --> w --> x_grad --> input for prev layer in backpropogation
        w_grad = x @ dl/dz
        b_grad = 1 @ dl/dz
        x_grad = w @ dl/dz --> return
        """
        activation_grad = derivative_prev_layer * self.derivative(self.outputs)
        self.w_grad = activation_grad.T @ self.inputs
        self.b_grad = np.sum(activation_grad, axis=0, keepdims=True)
        next_grad = activation_grad @ self.weights
        return next_grad