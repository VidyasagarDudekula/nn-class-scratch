import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(outputs):
    return outputs * (1 - outputs)


def softmax(x):
    exp_val = np.exp(x-np.max(x, axis=1, keepdims=True))
    return exp_val/np.sum(exp_val, axis=1, keepdims=True)


def softmax_derivative(outputs):
    return np.ones_like(outputs)


def ReLU(x):
    return np.maximum(0, x)


def ReLU_derivative(outputs):
    return np.where(outputs>0, 1, 0)


def linear(x):
    return x


def linear_derivative(outputs):
    return np.ones_like(outputs)


def RandomWeights(in_features: int, out_features: int):
    return np.random.randn(out_features, in_features) * 0.01


def XavierWeights(in_features: int, out_features: int):
    stddev = np.sqrt(2.0/(in_features + out_features))
    return np.random.randn(out_features, in_features) * stddev


def HeWeights(in_features: int, out_features: int):
    limit = np.sqrt(6.0/in_features)
    return np.random.uniform(-limit, limit, (out_features, in_features))


def MSELoss(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)


def CrossEntropyLoss(y_true, y_pred, target_value=0.99):
    # clip:-
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    mask = y_true == target_value
    logits = y_pred[mask].flatten()
    log_probs = np.log(logits)
    return -np.mean(log_probs)


def CrossEntropyLoss_derivative(y_true, y_pred, n):
    return (y_pred - y_true)/n


def MSELoss_derivative(y_true, y_pred, n):
    return 2.0 * (y_pred - y_true)/n