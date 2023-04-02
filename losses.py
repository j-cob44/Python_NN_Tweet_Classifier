# Losses.py - Loss Functions for Neural Networks derived from Common Loss Class
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np

# Common Loss Class
class Loss:
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculate regularization loss
    def regularization_loss(self):
        regularization_loss = 0 # 0 by default

        #Calculate regularization loss for all trainable layers
        for layer in self.trainable_layers:
            # L1 Weights regularization
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            # L2 Weights regularization
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
            # L1 Biases regularization
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            # L2 Biases regularization
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        
        return regularization_loss
    
    # Calculate data and regularization losses
    def calculate(self, output, y, *, include_regularization=False):
        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean losses
        data_loss = np.mean(sample_losses) # batch sample_losses

        # Accumulated sum of Losses and and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        # If just data loss, return
        if not include_regularization:
            return data_loss    

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # Calculate accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss, return
        if not include_regularization:
            return data_loss
        
        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss on new pass
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Categorical Cross-Entropy Loss Class
class Loss_CategoricalCrossEntropy(Loss):
    # Forward Pass
    def forward(self, y_prediction, y_true):
        # Numbers of samples in a batch
        samples = len(y_prediction)
        y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7) # Clip data to prevent division by 0

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_cofidences = y_prediction_clipped[range(samples), y_true] # Scaler values
        # Mask values - only for one-hot encoded values
        elif len(y_true.shape) == 2:
            correct_cofidences = np.sum(y_prediction_clipped * y_true, axis=1) # One-Hot Encoded vector

        # Loss
        negative_log_likelihoods = -np.log(correct_cofidences)
        return negative_log_likelihoods
    
    # Backward Pass
    def backward(self, dvalues, y_true):
        samples = len(dvalues) # total number of samples
        labels = len(dvalues[0]) # total number of labels/indexs in every samples

        # If labels are sparse, use one-hot encoding vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues # Derivative of inputs
        self.dinputs = self.dinputs / samples # Normalization

# Binary Cross-Entropy Loss Class
class Loss_BinaryCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Clip data to prevent division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error Loss Class
class Loss_MeanSquaredError(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error Loss Class
class Loss_MeanAbsoluteError(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        outputs = len(dvalues[0])

        # Calculate gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples