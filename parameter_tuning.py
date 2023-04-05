# Parameter_Tuning.py - Contains functions for automatically tuning the Neural Network to find the best parameters
# Jacob Burton 2023

import itertools
import numpy as np
import json
import multiprocessing
from joblib import Parallel, delayed
from model_actions import *
from model import *

# Save Parameters to JSON File
def save_parameters(parameters, filename="parameter_data/test.json"):
    with open(filename, "w") as f:
        json.dump(parameters, f, indent=4)

def run_model(parameters, X, y, X_val, y_val, set_num, total_sets):
    # Create Model from parameters
    model = create_model(
        hidden_layers=parameters["hidden_layers"], 
        neurons=parameters["neurons"], 
        dropout=parameters["dropout_rate"], 
        learning_rate=parameters["learning_rate"], 
        learning_decay=parameters["learning_decay"], 
        weight_regularizer_l1=parameters["weight_regularizer_l1"], 
        weight_regularizer_l2=parameters["weight_regularizer_l2"],
        bias_regularizer_l1=parameters["bias_regularizer_l1"],
        bias_regularizer_l2=parameters["bias_regularizer_l2"])
    
    # Print Parameters set that is running
    print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
        "[Set #" + f'{set_num}/{total_sets}' +
        ", Hidden Layers: " + f'{parameters["hidden_layers"]}' + 
        ", Neurons: " + f'{parameters["neurons"]}' + 
        ", Dropout Rate: " +  f'{parameters["dropout_rate"] * 100}%' + 
        ", Learning Rate: " +  f'{parameters["learning_rate"]}' +
        ", Learning Rate Decay: " + f'{parameters["learning_decay"]}' + ',\n\t'
        "Weight Regularization L1: " +  f'{parameters["weight_regularizer_l1"]}' +
        ", Bias Regularization L1: " +  f'{parameters["bias_regularizer_l1"]}' +
        ", Weight Regularization L2: " +  f'{parameters["weight_regularizer_l2"]}' +
        ", Bias Regularization L2: " +  f'{parameters["bias_regularizer_l2"]}' +
        "]")

    # Train Model
    model.train(X, y, iterations=parameters["iterations"], batch_size=parameters["batch_size"], print_every=0)

    # Get Accuracy
    validation_data = model.evaluate(X_val, y_val, print_summary=0)
    accuracy = validation_data["validation_accuracy"]

    # Print Finished Model Stats
    print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) + "[Set #" + f'{set_num}/{total_sets}' + ", Validation, " + 
          "Accuracy: " + f'{validation_data["validation_accuracy"]:.3f}' + " Loss: " + f'{validation_data["validation_loss"]:.3f}' + ']')

    return accuracy

# Grid Search for Hyperparameter Tuning
def perform_grid_search(param_file_path, X, y, X_val, y_val, 
    iterations=[10], 
    batch_size=[64],
    hidden_layers=[2], 
    neurons=[128], 
    dropouts=[0.1], 
    learning_rates=[0.01], 
    learning_decays=[1e-3], 
    weight_regularizers_l1=[0], 
    bias_regularizers_l1=[0],
    weight_regularizers_l2=[0.001], 
    bias_regularizers_l2=[0.001]):

    # Initialize best_parameter object
    best_parameters = {}

    # Split X and y by 25%
    training_full_length = len(X)
    training_split = int(training_full_length * 0.25)
    X = X[:training_split]
    y = y[:training_split]

    # Get start time
    start_time = time.time()

    # Hyperparameters Grid to search over
    param_grid = {
        "weight_regularizer_l1": weight_regularizers_l1, 
        "bias_regularizer_l1": bias_regularizers_l1,
        "weight_regularizer_l2": weight_regularizers_l2, 
        "bias_regularizer_l2": bias_regularizers_l2,
        "learning_decay": learning_decays, 
        "learning_rate": learning_rates, 
        "dropout_rate": dropouts, 
        "neurons": neurons, 
        "hidden_layers": hidden_layers, 
        "batch_size": batch_size,
        "iterations": iterations
    }

    # Generate all combinations of hyperparameters
    param_combinations = list(itertools.product(*[[(k, v) for v in param_grid[k]] for k in param_grid]))

    # Define Threads
    num_threads = (multiprocessing.cpu_count() // 2) # Use 1/2 of available threads | 87% cpu usage | ~42 hours with 12 threads 
    print("Number of Threads: " + str(num_threads))

    # Get Start Time
    start_time = time.time()

    # Run Grid Search in Parallel
    results = []

    try:
        results = Parallel(n_jobs=num_threads)(
            delayed(run_model)(dict(parameter_set), X, y, X_val, y_val, param_combinations.index(parameter_set), len(param_combinations)) for parameter_set in param_combinations)
    except KeyboardInterrupt:
        print("Keyboard Interrupt Detected, Stopping Search...")
        return

    # Find best set of hyperparameters
    best_parameters = param_combinations[results.index(max(results))]
    best_parameters = dict(best_parameters)

    # Get end time
    end_time = time.time()

    # Save Results
    stats = {
        "time_elapsed": end_time - start_time,
        "best_params": {
            "iterations": best_parameters["iterations"],
            "batch_size": best_parameters["batch_size"],
            "hidden_layers": best_parameters["hidden_layers"], 
            "neurons": best_parameters["neurons"], 
            "dropout": best_parameters["dropout_rate"], 
            "learning_rate": best_parameters["learning_rate"], 
            "learning_decay": best_parameters["learning_decay"], 
            "weight_regularizer_l1": best_parameters["weight_regularizer_l1"], 
            "weight_regularizer_l2": best_parameters["weight_regularizer_l2"],
            "bias_regularizer_l1": best_parameters["bias_regularizer_l1"],
            "bias_regularizer_l2": best_parameters["bias_regularizer_l2"]
        }
    }

    # Save Best Parameters
    save_parameters(stats, filename=param_file_path)

    # End grid search