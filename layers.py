# Layers.py - Neural Network Layers Classes
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np

# Dense Layer Class
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # Set Regularization Strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    # Backward pass
    def backward(self, dvalues):
        # Gradient on Parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on Regularizations
        # L1 on Weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        
        # L2 on Weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on Biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        
        # L2 on Biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
        
        # Gradient on Output Values
        self.dinputs = np.dot(dvalues, self.weights.T)
    
    # Retrieve Layer Parameters from Layer
    def get_parameters(self):
        return self.weights, self.biases
    
    # Set Layer Parameters
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout Layer Class
class Layer_Dropout:
    # Layer Initialization
    def __init__(self, rate):
        # Invert Dropout Rate so 0.1 needs a success rate of 0.9
        self.rate = 1 - rate
    
    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return
        
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        # Apply Mask to output
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on output values
        self.dinputs = dvalues * self.binary_mask

# Input Layer Class
class Layer_Input:
    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs