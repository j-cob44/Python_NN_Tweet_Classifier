# Parameter_Tuning.py - Contains functions for automatically tuning the Neural Network to find the best parameters
# Jacob Burton 2023

import numpy as np
import json
from model_actions import *
from model import *

# Save Parameters to JSON File
def save_parameters(parameters, filename="parameter_data/test.json"):
    with open(filename, "w") as f:
        json.dump(parameters, f, indent=4)

# Grid Search for Hyperparameter Tuning
def perform_grid_search(param_file_path, X, y, X_val, y_val, 
    iterations=10, batch_size=64,
    hidden_layers=[2], 
    neurons=[128], 
    dropouts=[0.1], 
    learning_rates=[0.01], 
    learning_decays=[1e-3], 
    weight_regularizers_l1=[0.01], 
    bias_regularizers_l1=[0.01],
    weight_regularizers_l2=[0.01], 
    bias_regularizers_l2=[0.01]):

    # Initialize Variables
    current_parameter_set = 1
    best_accuracy = 0
    best_parameters = {}

    # Get start time
    start_time = time.time()

    # Initialize Model
    model = Model()

    # Iterate through all possible combinations of parameters
    for hidden_layer in hidden_layers:
        for neuron in neurons:
            for dropout in dropouts:
                for learning_rate in learning_rates:
                    for learning_decay in learning_decays:
                        for weight_regularizer_l1 in weight_regularizers_l1:
                            for bias_regularizer_l1 in bias_regularizers_l1:
                                for weight_regularizer_l2 in weight_regularizers_l2:
                                    for bias_regularizer_l2 in bias_regularizers_l2:
                                        # Print Progress
                                        print(time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
                                              " [Set #" + f'{current_parameter_set}' +
                                              ", Hidden Layers: " + f'{hidden_layer}' + 
                                              ", Neurons: " + f'{neuron}' + 
                                              ", Dropout Rate: " +  f'{dropout}%' + 
                                              ", Learning Rate: " +  f'{learning_rate}' +
                                              ", Learning Rate Decay: " + f'{learning_decay}' + ',\n\t'
                                              "Weight Regularization L1: " +  f'{weight_regularizer_l1}' +
                                              ", Bias Regularization L1: " +  f'{weight_regularizer_l2}' +
                                              ", Weight Regularization L2: " +  f'{weight_regularizer_l1}' +
                                              ", Bias Regularization L2: " +  f'{bias_regularizer_l2}' +
                                              "]")

                                        # Create Model
                                        model = create_model(
                                            hidden_layers=hidden_layer, 
                                            neurons=neuron, 
                                            dropout=dropout, 
                                            learning_rate=learning_rate, 
                                            learning_decay=learning_decay, 
                                            weight_regularizer_l1=weight_regularizer_l1, 
                                            weight_regularizer_l2=weight_regularizer_l2,
                                            bias_regularizer_l1=bias_regularizer_l1,
                                            bias_regularizer_l2=bias_regularizer_l2)
                                        
                                        # Train Model
                                        model.train(X, y, iterations=iterations, batch_size=batch_size, print_every=0, validation_data=(X_val, y_val))

                                        # Get Accuracy
                                        validation_data = model.evaluate(X_val, y_val)
                                        accuracy = validation_data["validation_accuracy"]

                                        # Check if Accuracy is Best
                                        if accuracy > best_accuracy:
                                            best_accuracy = accuracy
                                            best_parameters = {
                                                "time_of_search": 0,
                                                "hidden_layers": hidden_layer,
                                                "neurons": neuron,
                                                "dropout": dropout,
                                                "learning_rate": learning_rate,
                                                "learning_decay": learning_decay,
                                                "weight_regularizer_l1": weight_regularizer_l1,
                                                "bias_regularizer_l1": bias_regularizer_l1,
                                                "weight_regularizer_l2": weight_regularizer_l2,
                                                "bias_regularizer_l2": bias_regularizer_l2
                                            }

                                        current_parameter_set+=1
    
    # Get end time
    end_time = time.time()

    total_time = end_time - start_time
    best_parameters["time_of_search"] = total_time

    # Print stats
    print("Time Elapsed: ", f'{total_time:.2f}')
    print("Best Hyperparameters: ", best_parameters)
    print("Best Accuracy: ", best_accuracy)

    # Save Best Parameters
    save_parameters(best_parameters, filename=param_file_path)

    # End grid search