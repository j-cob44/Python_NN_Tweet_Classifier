# Optimizers.py - Different Optimizer Classes for Neural Networks
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np

# Stochastic Gradient Descent Object with Momentum and Learning Rate Decay
# Uses a Global Learning Rate and a Global Decay Rate
class Optimizer_SGD:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iterations = 0

    # Call once before any layer is updated
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update weights and biases of a layer
    def update_params(self, layer):
        # Using Momentum
        if self.momentum:
            # If Weight Momentums has not been created yet
            if not hasattr(layer, 'weight_momentums'):
                # Create weight and bias momentum arrays
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Calculate Weights with Momentum
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Calculate Biases with Momentum
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            # Original SGD Updates
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after every layer has updated
    def post_update_params(self):
        self.iterations += 1

# Adaptive Gradient Descent Object with Rate Decay using a Epsilon hyperparameter
# Uses Per-Parameter Adaptive Learning Rate
class Optimizer_AdaGrad:
    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    # Call once before any layer is updated
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update weights and biases of a layer
    def update_params(self, layer):
        # Create Cache Arrays for history of gradients
        if not hasattr(layer, 'weight_cache'):
            # Create weight and bias cache arrays
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with Gradients squared
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after every layer has updated
    def post_update_params(self):
        self.iterations += 1

# Root Mean Square Propagation Object with a cache memory decay rate
# Uses Per-Parameter Adaptive Learning Rate
class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0

    # Call once before any layer is updated
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update weights and biases of a layer
    def update_params(self, layer):
        # Create Cache Arrays for history of gradients
        if not hasattr(layer, 'weight_cache'):
            # Create weight and bias momentum arrays
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache with Gradients squared
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    # Call once after every layer has updated
    def post_update_params(self):
        self.iterations += 1

# Adaptive Momentum Optimizer, currently most widely-used optimizer
# Uses Per-Parameter Adaptive Learning Rate with Momentum
class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iterations = 0

    # Call once before any layer is updated
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    # Update weights and biases of a layer
    def update_params(self, layer):
        # Create Cache Arrays for history of gradients
        if not hasattr(layer, 'weight_cache'):
            # Create weight and bias momentum and cache arrays
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update Momentums
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Calculate Corrected Momentums
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1**(self.iterations + 1))

        # Update cache with Gradients squared
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Calculate Corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2**(self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after every layer has updated
    def post_update_params(self):
        self.iterations += 1