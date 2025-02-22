from neural_network import NeuralNetwork
import numpy as np

# Example: XOR problem
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
y = np.array([[0, 1, 1, 0]])

# Create a neural network with architecture: 2 inputs, 4 hidden neurons, 1 output
nn = NeuralNetwork([2, 4, 1])

# Train the network
output = nn.train(X, y, epochs=10000)

# Test the network
print("Predictions:")
print(output)
print("\nExpected:")
print(y)
