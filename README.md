# Neural Network from Scratch

This project implements a fully functional Multi-Layer Perceptron (MLP) neural network from scratch using Python and NumPy. It demonstrates the core concepts of deep learning, including forward propagation, backpropagation, and gradient descent, without relying on high-level deep learning frameworks like PyTorch or TensorFlow.

## 🚀 Overview

The goal of this project is to build and train a neural network to recognize handwritten digits from the MNIST dataset. The implementation is modular, with separate classes for the network model, layers, and utility functions.

**Key Achievements:**
-   **Accuracy:** Achieved **97.91%** accuracy on the MNIST test dataset.
-   **Test Loss:** Final test loss of **0.0784**.
-   **Pure Implementation:** Built entirely with NumPy for matrix operations.

## 🧠 Architecture

The neural network consists of an input layer, two hidden layers, and an output layer.

-   **Input Layer:** 784 neurons (corresponding to 28x28 pixel images).
-   **Hidden Layer 1:** 150 neurons with **ReLU** activation.
-   **Hidden Layer 2:** 150 neurons with **ReLU** activation.
-   **Output Layer:** 10 neurons with **Softmax** activation (representing digits 0-9).

### Model Configuration
-   **Optimizer:** Stochastic Gradient Descent (SGD)
-   **Loss Function:** Cross Entropy Loss
-   **Batch Size:** 64
-   **Training Steps:** 20,000
-   **Learning Rate:** 0.1

## 📂 Implementation Details

The codebase is structured into several key files:

-   **`nn.py`**: The main driver script. It handles data loading, model initialization, the training loop, testing, and plotting results.
-   **`mlp.py`**: Contains the `MLP` class, which acts as a container for layers and manages the forward and backward passes for the entire network.
-   **`dense_layer.py`**: Implements individual fully connected layers, including weight initialization and activation functions.
-   **`functions.py`**: Provides utility functions such as loss calculations and derivatives.

## 📊 Results & Performance

The model was trained for 20,000 iterations. Below is the training loss curve, showing the convergence of the model over time.

![Training Loss Graph](training_loss_graph.png)

### Final Statistics
The final evaluation on the test set yielded the following results (from `stats.json`):

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | **97.91%** |
| **Test Loss** | **0.0784** |

## 🛠️ Usage

1.  **Dependencies**: Ensure you have Python installed along with `numpy` and `matplotlib`.
    ```bash
    pip install numpy matplotlib
    ```

2.  **Run Training**: Execute the `nn.py` script to start training the model.
    ```bash
    python nn.py
    ```
    This script will:
    -   Load the MNIST dataset.
    -   Train the MLP network.
    -   Save the trained model to `model_mnist.npz`.
    -   Save performance stats to `stats.json`.
    -   Display the training loss graph.

3.  **Output**: The script prints the training progress and final test accuracy to the console.

## 📝 Note
The data loading in `nn.py` assumes `datasets/mnist_train.csv` and `datasets/mnist_test.csv` exist. Ensure these files are present in the `datasets` directory.
