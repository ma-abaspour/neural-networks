import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases with random values
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i]))
            self.biases.append(np.random.randn(layer_sizes[i+1], 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_propagation(self, x):
        self.activations = [x]
        current_activation = x
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, current_activation) + b
            current_activation = self.sigmoid(z)
            self.activations.append(current_activation)
            
        return self.activations[-1]
    
    def backward_propagation(self, x, y, learning_rate=0.1):
        m = x.shape[1]
        delta = self.activations[-1] - y
        
        for layer in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(delta, self.activations[layer].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            if layer > 0:
                delta = np.dot(self.weights[layer].T, delta) * self.sigmoid_derivative(self.activations[layer])
            
            self.weights[layer] -= learning_rate * dW
            self.biases[layer] -= learning_rate * db
    
    def train(self, X, y, epochs, learning_rate=0.1):
        for _ in range(epochs):
            output = self.forward_propagation(X)
            self.backward_propagation(X, y, learning_rate)
        return output
