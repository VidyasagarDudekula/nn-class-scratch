import numpy as np
from mlp import MLP
from dense_layer import DenseLayer
import functions as F
import numpy as np
import matplotlib.pyplot as plt
from data_processing import preprocess_line, preprocess_input, preprocess_output
import json


def train_test_split(xs, ys, test_split=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    idx = np.arange(0, xs.shape[0])
    np.random.shuffle(idx)
    xs = xs[idx]
    ys = ys[idx]
    train_limit = int(xs.shape[0]*(1-0.2))
    xs_train, xs_test = xs[:train_limit], xs[train_limit:]
    ys_train, ys_test = ys[:train_limit], ys[train_limit:]
    return xs_train, ys_train, xs_test, ys_test


def get_data(file_name = "datasets/mnist_train.csv"):
    data_file = open(file_name, 'r')
    data_lines = data_file.readlines()[1:]
    data_file.close()
    xs, ys = [], []
    for line in data_lines:
        line_data = preprocess_line(line)
        input_list = preprocess_input(line_data[0])
        target_list = preprocess_output(line_data[1])
        xs.append(input_list)
        ys.append(target_list)
    return xs, ys


xs, ys = get_data()
xs_train = np.array(xs, ndmin=2)
ys_train = np.array(ys, ndmin=2)


model = MLP(
    DenseLayer(784, 150, "relu"),
    # DenseLayer(200, 150, "relu"),
    DenseLayer(150, 150, "relu"),
    DenseLayer(150, 10, "softmax")
)

stepi = []
lossi = []

batch_size = 64

learing_rate = 0.1

for step in range(20000):
    idx = np.random.randint(0, xs_train.shape[0], size=batch_size)
    # print(idx)
    batch_xs = xs_train[idx]
    batch_ys = ys_train[idx]
    model.zero_grad()
    outputs = model(batch_xs)
    loss = F.CrossEntropyLoss(batch_ys, outputs, target_value=1.0)
    stepi.append(step)
    lossi.append(loss)
    loss_grad = F.CrossEntropyLoss_derivative(batch_ys, outputs, batch_size)
    model.backward(loss_grad)
    model.step(learning_rate=learing_rate)


print(f"Train loss:- ", min(lossi))

xs_test, ys_test = get_data(file_name="datasets/mnist_test.csv")
xs_test = np.array(xs_test, ndmin=2)
ys_test = np.array(ys_test, ndmin=2)

model.zero_grad()
test_outputs = model(xs_test)
loss_test = F.CrossEntropyLoss(ys_test, test_outputs, target_value=1.0)
print(f"Loss test:- {loss_test}")

predictions = np.argmax(test_outputs, axis=1)
actuals = np.argmax(ys_test, axis=1)

accuracy = np.mean(predictions == actuals) * 100
print(f"Final Test Accuracy: {accuracy:.2f}%")

stats = {
    "test_loss": loss_test,
    "test_accuracy": accuracy,
}
with open('stats.json', 'w') as json_file:
    json.dump(stats, json_file, indent=4)

model.save('model_mnist.npz')

plt.plot(stepi, lossi)
plt.show()
