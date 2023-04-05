# Model.py - Model Class for a Neural Network
# Jacob Burton 2023
# With help from NNFS.io Book by Harrison Kinsley and Daniel Kukiela

import numpy as np
import time
import pickle
import copy

from accuracies import *
from activations import *
from layers import *
from losses import *
from optimizers import *

# Neural Network Model Class
class Model:
    def __init__(self):
        # Create a list of layers
        self.layers = []

        # Softmax Classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the Network
    def add(self, layer):
        self.layers.append(layer)
    
    # Set Loss, Optimizer, and Accuracy Parameters of the Neural Network
    def set(self, *, loss, optimizer, accuracy):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy
    
    # Finalize the Model
    def finalize(self):
        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the Network Objects
        layer_count = len(self.layers)

        # Initialize a list of trainable layers
        self.trainable_layers = []

        # Iterate the object list
        for i in range(layer_count):
            # If it is the first layer, set input layer as previous layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            # Layers other than first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # Last layer, next object is loss
            # Save reference to last object
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # Find if Layer is trainable
            # Trainable layers have weights
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            if self.loss is not None:
                # Update loss object with trainable layers
                self.loss.remember_trainable_layers(self.trainable_layers)
        
        # If output activation is softmax and loss function is categorical cross-entropy
        # Create an object of combined activation and loss for efficiency
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            # Create an object of combined activation and loss
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()

    # Forward Pass of All Layers
    def forward(self, X, training):
        # Forward Pass of Input Layer
        self.input_layer.forward(X, training)

        # Forward Pass of all other layers
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        
        # Return the output of the last layer
        return layer.output
    
    # Backward Pass of All Layers
    def backward(self, output, y):
        # If softmax classifier is used
        if self.softmax_classifier_output is not None:
            # First call backward method on combined activation/loss methood
            self.softmax_classifier_output.backward(output, y)

            # Set dinputs of this object to dinputs of combined object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backwards pass on all layers but last
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # Else,
        # First, call backward method on loss
        self.loss.backward(output, y)

        # Call backward pass on all layer objects in reverse order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    # Train the Model
    def train(self, X, y, *, iterations=1, batch_size=None, print_every=1, validation_data=None):
        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not set
        training_steps = 1

        # If validation data is provided
        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            training_steps = len(X) // batch_size # Rounded Down
            
            # If there are some remaining data, add 1 more step
            if training_steps * batch_size < len(X):
                training_steps += 1
            
            if validation_data is not None:
                validation_steps = len(validation_data[0]) // batch_size # Rounded Down

                # If there are some remaining data, add 1 more step
                if validation_steps * batch_size < len(validation_data[0]):
                    validation_steps += 1

        # Training data for analysis
        analysis_data = {
            "training_accuracy": None,
            "validation_accuracy": None,
            "validation_loss": None,
            "accuracy_history": [],
            "loss_history": [],
            "learning_rate_history": []
        }

        # Training Loop
        for iteration in range(1, iterations+1):
            # Print Iteration Number
            print(time.strftime("[%H:%M:%S]", time.localtime(time.time())),"Iteration: ", iteration)

            # Reset Accumulated Values in Loss and Accuracy Objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over Steps
            for step in range(training_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform Forward Pass of all layers
                output = self.forward(batch_X, training=True)

                # Calculate Loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get Predictions and Calculate Accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Preform Backward Pass of all layers
                self.backward(output, batch_y)

                # Last step for iteration, optimize (update) weights and biases of trainable layers
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Gather data for Analysis
                analysis_data["accuracy_history"].append(accuracy)
                analysis_data["loss_history"].append(loss)
                analysis_data["learning_rate_history"].append(self.optimizer.current_learning_rate)

                # Print a summary for the step
                if not step % print_every or step == training_steps - 1:
                    print(time.strftime("[%H:%M:%S]", time.localtime(time.time())), " Step:", step, " Accuracy: ", f'{accuracy:.3f}', " Loss: ", f'{loss:.3f}', "\n\t", "(Data_Loss: ", f'{data_loss:.3f}', "+ Reg_Loss: ", f'{regularization_loss:.3f})', " LearningRate: ", self.optimizer.current_learning_rate )

            # Calculate and Print Iteration Loss and Accuracy
            iteration_data_loss, iteration_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            iteration_loss = iteration_data_loss + iteration_regularization_loss
            iteration_accuracy = self.accuracy.calculate_accumulated()

            # Print a summary for the iteration
            print(time.strftime("[%H:%M:%S]", time.localtime(time.time())), "Training Overall: ", " Accuracy: ", f'{iteration_accuracy:.3f}', " Loss: ", f'{iteration_loss:.3f}', "\n\t", "(Data_Loss: ", f'{iteration_data_loss:.3f}', " Reg_Loss: ", f'{iteration_regularization_loss:.3f})', " LearningRate: ", self.optimizer.current_learning_rate )

        analysis_data["training_accuracy"] = self.accuracy.calculate_accumulated()

        if validation_data is not None:
            print("Validating Model")
            validation_data = self.evaluate(*validation_data, batch_size=batch_size)

        analysis_data["validation_accuracy"] = validation_data["validation_accuracy"]
        analysis_data["validation_loss"] = validation_data["validation_loss"]
        
        return analysis_data
    
    # Evaluate the Model with a given dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):
        # Default value if batch size is not set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size # Rounded Down

            # If there are some remaining data, add 1 more step
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        
        # Reset accumulated values in loss and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform Forward Pass of all layers
            output = self.forward(batch_X, training=False)

            # Calculate Loss
            self.loss.calculate(output, batch_y)

            # Get Predictions and calculate accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Calculate and print Validation Loss and Accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print Validation Summary
        print(time.strftime("[%H:%M:%S]", time.localtime(time.time())), "Validation, ", " Accuracy: ", f'{validation_accuracy:.3f}', " Loss: ", f'{validation_loss:.3f}')

        validation_data = {
            "validation_accuracy": validation_accuracy,
            "validation_loss": validation_loss
        }
        return validation_data

    # Make a Prediction on Given Data
    def predict(self, X, *, batch_size=None):
        # Default value if batch size is not set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size # Rounded Down

            # If there are some remaining data, add 1 more step
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        # Model Outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform Forward Pass of all layers
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to list of predictions
            output.append(batch_output)
        
        # Stack and return results
        return np.vstack(output)
    
    # Get Parameters of the Model
    def get_parameters(self):
        # Create a list of parameters
        parameters = []

        # Iterate over all trainable layers and retrieve parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters
    
    # Set Parameters of the Model
    def set_parameters(self, parameters):
        # Iterate over all trainable layers and set parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Save Parameters to File
    def save_parameters(self, path):
        # Save parameters to a file in binary-write mode
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    # Load Parameters from File
    def load_parameters(self, path):
        # Load parameters from a file in binary-read mode
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Save Entire Model to File
    def save(self, path):
        # Make a deep copy of the current model
        model = copy.deepcopy(self)

        # Remove accumulated values from loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # Remove inputs, outputs, and dinput properties from all layers
        for layer in model.layers:
            for property in ['inputs', 'outputs', 'dinputs', 'doutputs']:
                layer.__dict__.pop(property, None)
        
        # Save the model to a file in binary-write mode
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    # Load Entire Model from File
    @staticmethod
    def load(path):
        # Open file in binary-read mode
        with open(path, 'rb') as f:
            # Load the model
            model = pickle.load(f)
        
        return model