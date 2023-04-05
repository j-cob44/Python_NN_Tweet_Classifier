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
    weight_regularizers_l1=[0], 
    bias_regularizers_l1=[0],
    weight_regularizers_l2=[0.001], 
    bias_regularizers_l2=[0.001]):

    # Initialize Variables
    current_parameter_set = 1
    total_parameter_sets = len(hidden_layers) * len(neurons) * len(dropouts) * len(learning_rates) * len(learning_decays) * len(weight_regularizers_l1) * len(bias_regularizers_l1) * len(weight_regularizers_l2) * len(bias_regularizers_l2)
    best_accuracy = 0
    best_parameters = {}

    # Get start time
    start_time = time.time()



    # Iterate through all possible combinations of parameters
    for bias_regularizer_l2 in bias_regularizers_l2:
        for weight_regularizer_l2 in weight_regularizers_l2:
            for bias_regularizer_l1 in bias_regularizers_l1:
                for weight_regularizer_l1 in weight_regularizers_l1:
                    for learning_decay in learning_decays:
                        for learning_rate in learning_rates:
                            for dropout in dropouts:
                                for neuron in neurons:
                                    for hidden_layer in hidden_layers:
                                        # Print Progress
                                        print(time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
                                              " [Set #" + f'{current_parameter_set}/{total_parameter_sets}' +
                                              ", Hidden Layers: " + f'{hidden_layer}' + 
                                              ", Neurons: " + f'{neuron}' + 
                                              ", Dropout Rate: " +  f'{dropout*10}%' + 
                                              ", Learning Rate: " +  f'{learning_rate}' +
                                              ", Learning Rate Decay: " + f'{learning_decay}' + ',\n\t'
                                              "Weight Regularization L1: " +  f'{weight_regularizer_l1}' +
                                              ", Bias Regularization L1: " +  f'{bias_regularizer_l1}' +
                                              ", Weight Regularization L2: " +  f'{weight_regularizer_l2}' +
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
                                        analysis_data = model.train(X, y, iterations=iterations, batch_size=batch_size, print_every=0, validation_data=(X_val, y_val))

                                        # Get Accuracy
                                        accuracy = analysis_data["validation_accuracy"]

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