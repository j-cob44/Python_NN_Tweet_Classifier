# Activations.py - Different types on Neural Network Activation Functions
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np
from scipy.special import expit

# Rectified Linear Activation Class (ReLU)
class Activation_ReLU:
    # Forward Pass
    def forward(self, inputs, training):
        self.inputs = inputs # Save input values
        self.output = np.maximum(0, inputs)

    # Backward Pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
    
    # Calculate predictions for ReLU outputs
    def predictions(self, outputs):
        return outputs

# Softmax Activation Class
class Activation_Softmax:
    # Forward Pass
    def forward(self, inputs, training):
        self.inputs = inputs # Save input values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # axis and keepdims used for enabling batching, unnormalized probability
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True) # axis and keepdims used for enabling batching, Output are normalized probability
    
    # Backward Pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) # Create array of size of dvalues

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1) # Flatten output array
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) # Calculate Jacobian matrix of the output and transposed output
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues) # Calculate sample-wise gradient

    # Calculate predictions for Softmax
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Combined Softmax Activation and Cross-Entropy Loss Class
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    # Backward Pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues) # Total number of sample_losses
        self.dinputs = dvalues.copy() # Copy to working variable
        self.dinputs[range(samples), y_true] -= 1 # Calculate gradient
        self.dinputs = self.dinputs / samples # Normalization of gradient

# Sigmoid Activation Class - Used to predict binary values (0 or 1)
class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs # Save input values
        self.output = expit(inputs) #1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for Sigmoid
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
# Linear Activation Class - Used as an output for regression problems where the goal is to predict a continuous value
class Activation_Linear:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

    # Calculate predictions for Linear
    def predictions(self, outputs):
        return outputs
    
# Tanh Activation Class - Tanh is used as a smooth function that can be easily differentiated since the output range is between -1 and 1
class Activation_Tanh:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)

    # Calculate predictions for Tanh
    def predictions(self, outputs):
        return outputs