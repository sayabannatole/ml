# Taken the example form OpenAI models while learning the ML
import numpy as np

# Inputs (X), Weights (W), and Bias (b)
X = np.array([0.5, 0.3])  # Example input values
W = np.array([0.7, 0.9])  # Assigned weights
b = 0.2  # Bias term

# Activation Function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Compute Neuron Output
linear_output = np.dot(W, X) + b  # WX + b
output = sigmoid(linear_output)  # Apply activation function

print("Neuron Output:", output)
