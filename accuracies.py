# Accuracies.py - Accuracy Classes for Neural Networks
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np

# Common Accuracy Class
class Accuracy:
    # Calculates accuracy
    def calculate(self, predictions, y):
        # Get Comparison results
        comparisons = self.compare(predictions, y)

        # Calculate accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy
    
    # Calculates accumulated accuracy
    def calculate_accumulated(self, *, include_regularization=False):
        # Calculate accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        return accuracy

    # Reset variables for accumulated accuracy on new pass
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

# Accuracy calculation class for regression model
class Accuracy_Regression(Accuracy):
    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculate Precision Value, based on passed in ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
    
    # Compare predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    
# Accuracy calculation class for classification model
class Accuracy_Categorical(Accuracy):
    # Initialization not needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y