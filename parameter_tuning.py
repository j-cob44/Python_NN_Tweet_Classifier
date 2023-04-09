# Parameter_Tuning.py - Contains functions for automatically tuning the Neural Network to find the best parameters
# Jacob Burton 2023

import itertools
import math
import random
import numpy as np
import json
import multiprocessing
from joblib import Parallel, delayed
from bayes_opt import BayesianOptimization
from model_actions import *
from model import *

# Save Parameters to JSON File
def save_parameters(parameters, filename="parameter_data/test.json"):
    with open(filename, "w") as f:
        json.dump(parameters, f, indent=4)

def run_model(parameters, X, y, X_val, y_val, set_num, total_sets, display_data=True):
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
    if display_data:
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
    loss = validation_data["validation_loss"]

    # Print Finished Model Stats
    if display_data:
        print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) + "[Set #" + f'{set_num}/{total_sets}' + ", Validation, " + 
            "Accuracy: " + f'{validation_data["validation_accuracy"]:.3f}' + " Loss: " + f'{validation_data["validation_loss"]:.3f}' + ']')

    return (accuracy, parameters, loss)

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
            delayed(run_model)(dict(parameter_set), X, y, X_val, y_val, (param_combinations.index(parameter_set)+1), len(param_combinations)) for parameter_set in param_combinations)
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

# Random Search for Hyperparameter Tuning 
def random_search(param_file_path, X, y, X_val, y_val, search_iterations=110, 
    training_iterations=[10], 
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

    # Initialize variables
    best_parameters = None
    best_accuracy = float('-inf')

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
        "iterations": training_iterations
    }

    # Generate all combinations of hyperparameters
    param_space = list(itertools.product(*[[(k, v) for v in param_grid[k]] for k in param_grid]))
    
    # Split X and y by 25%
    training_full_length = len(X)
    training_split = int(training_full_length * 0.25)
    X = X[:training_split]
    y = y[:training_split]

    # Get start time
    start_time = time.time()

    # Define Threads
    num_threads = (multiprocessing.cpu_count() // 2) # Use 1/2 of available threads
    print("Number of Threads: " + str(num_threads))

    batches = search_iterations // num_threads
    # Round batches up to nearest integer
    iterations_run = 0

    # Run Random Search - Multithreaded
    results = []
    for i in (range(batches + 1)):
        print("\n[Batch #" + str(i) + "/" + str(batches) + "]")

        # Initialize variables
        random_param_sets = []
        batch_results = []

        # Get a n random hyperparameters from the parameter space
        if (iterations_run + num_threads < search_iterations):
            random_param_sets = random.choices(param_space, k=num_threads) # do full batch
        else:
            random_param_sets = random.choices(param_space, k=(search_iterations - iterations_run)) # do partial batch

        iterations_run += len(random_param_sets)
        
        # Run the model with the sampled hyperparameters
        batch_results = Parallel(n_jobs=num_threads)(
            delayed(run_model)(dict(parameter_set), X, y, X_val, y_val, (j+1), len(random_param_sets)) for j, parameter_set in enumerate(random_param_sets))
        
        # Remove used hyperparameters from the parameter space
        param_space = [x for x in param_space if x not in random_param_sets]

        # Add batch results to results
        results.extend(batch_results)

        print("\n[Batch #" + str(i) + " completed.]")

    print("\n[Random search completed.]")
    
    # Search Finished, Process Results
    best_parameters = None
    best_accuracy = None
    for result in results:
        score = result[0]
        if best_accuracy is None or score > best_accuracy:
            best_parameters = result[1]
            best_accuracy = score
    
    # Get end time
    end_time = time.time()

    # Save Results
    stats = {
        "time_elapsed": end_time - start_time,
        "best_accuracy": best_accuracy,
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

    # End random search

# Bayesian Search Optimization function
# search_iterations = ~10-20 times the number of parameters being optimized as a starting point.
def bayesian_search(param_file_path, X, y, X_val, y_val, 
    search_iterations=120, acq_func='ucb', kappa=2.576, sigma_noise=1e-6,
    target='accuracy',
    param_space = {
        "weight_regularizer_l1": (0, 0), 
        "bias_regularizer_l1": (0, 0),
        "weight_regularizer_l2": (0, 5e-07), 
        "bias_regularizer_l2": (0, 5e-07),
        "learning_decay": (1e-7, 1e-4), 
        "learning_rate": (0.0001, 0.1), 
        "dropout_rate": (0, 0.5), 
        "neurons": (32, 512), 
        "hidden_layers": (1, 3), 
        "activation_function": (int(0), int(3)), # (0, 1] = relu, (1, 2] = tanh, (2, 3] = sigmoid
        # Training Parameters
        "batch_size": (int(32), int(256)),
        "training_iterations": (int(8), int(32))
    }):

    # Define the search bounds for each parameter
    bounds = {}
    for key, value in param_space.items():
        # Order the bounds from lower to upper
        if value[0] < value[1]:
            bounds[key] = (value[0], value[1])
        else:
            bounds[key] = (value[1], value[0])

    # Slice X and y by 33%
    training_full_length = len(X)
    training_split = int(training_full_length * 0.4)
    X = X[:training_split]
    y = y[:training_split] 

    # Initialize variables for tracking progress
    i = 0;
    init_searches = len(bounds.keys())
    searches = search_iterations + init_searches
    best_score = 0

    # Initialize the object function to minimize, which is the negative accuracy
    def objective(**params):
        # Get iteration counts
        nonlocal i;
        nonlocal init_searches
        nonlocal searches;
        nonlocal best_score;
        nonlocal target;

        if i == init_searches + 1:
            print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
                  "[Initial parameter searches completed, beginning Bayesian Optimization.]")

        # Get activation function
        activation_value = int(math.ceil(params["activation_function"])) # 0-1 = relu, 1-2 = tanh, 2-3 = sigmoid, equal spacing

        if activation_value == 1:
            activation = "relu"
        elif activation_value == 2:
            activation = "tanh"
        elif activation_value == 3:
            activation = "sigmoid"
        else:
            raise Exception("Invalid activation function value: " + str(activation_value))

        # Convert parameters to dictionary
        param_set = {
            "weight_regularizer_l1": params["weight_regularizer_l1"], 
            "bias_regularizer_l1": params["bias_regularizer_l1"],
            "weight_regularizer_l2": params["weight_regularizer_l2"], 
            "bias_regularizer_l2": params["bias_regularizer_l2"],
            "learning_decay": params["learning_decay"], 
            "learning_rate": params["learning_rate"], 
            "dropout_rate": params["dropout_rate"], 
            "neurons": int(math.ceil((params["neurons"]))), # (31, 32] = 32, (32, 33] = 33, etc.
            "hidden_layers": int(math.ceil((params["hidden_layers"]))), # (0, 1] = 1, (1, 2] = 2, (2, 3] = 3, etc.
            "batch_size": int(math.ceil((params["batch_size"]))), # (31, 32] = 32, (63, 64] = 64, etc.
            "iterations": int(math.ceil((params["training_iterations"]))),  # (7, 8] = 8, (15, 16] = 16, etc.
            "activation": activation
        }

        # 
        print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
            "[Set: "+ f'{i}' + "/" + f'{searches}' +
            ": Training Iterations: " + f'{param_set["iterations"]}' +
            ", Batch Size: " + f'{param_set["batch_size"]}' +
            ", Activation Function: " + activation.upper() +
            ", \n\tHidden Layers: " + f'{param_set["hidden_layers"]}' + 
            ", Neurons: " + f'{param_set["neurons"]}' + 
            ", Dropout Rate: " +  f'{param_set["dropout_rate"] * 100}%' + 
            ", Learning Rate: " +  f'{param_set["learning_rate"]}' +
            ", Learning Rate Decay: " + f'{param_set["learning_decay"]}' +
            ", \n\tWeight Regularization L1: " +  f'{param_set["weight_regularizer_l1"]}' +
            ", Bias Regularization L1: " +  f'{param_set["bias_regularizer_l1"]}' +
            ", Weight Regularization L2: " +  f'{param_set["weight_regularizer_l2"]}' +
            ", Bias Regularization L2: " +  f'{param_set["bias_regularizer_l2"]}' +
            "]")

        # Train Model
        results = run_model(param_set, X, y, X_val, y_val, 1, 1, display_data=False)

        # Get score
        if target == "accuracy":
            score = results[0] # Score is for maximizing accuracy
        elif target == "loss":
            score = -results[2] # Score is for minimizing loss
        elif target == "both": 
            score = results[0] - results[2] # Score is for minimizing loss and maximizing accuracy
        else:
            raise Exception("Invalid target value: " + target)

        # if score is better than best score, print in pink
        if score > best_score and i > 0:
            print("\033[95m" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
                "[Results: " +
                "Score: " + f'{score}' +
                ", Validation Accuracy: " + f'{results[0] * 100}%' +
                ", Validation Loss: " + f'{results[2]}' + "]" + "\033[0m"
            )
            best_score = score
        else:
            # Print Results
            print(time.strftime("[%H:%M:%S]", time.localtime(time.time())) +
                "[Results: " +
                "Score: " + f'{score}' +
                ", Validation Accuracy: " + f'{results[0] * 100}%' +
                ", Validation Loss: " + f'{results[2]}' + "]"
            )

        # Increment iteration count
        i += 1

        return score

    
    # Get Start time
    start_time = time.time()
    print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) + "[Bayesian Search Started]")

    # Initialize Bayesian optimizer. From package BayesianOptimization by Fernando Nogueira at https://github.com/fmfn/BayesianOptimization
    optimizer = BayesianOptimization(f=objective, pbounds=bounds, verbose=0, random_state=1)

    # Optimize using initial points of length(parameters)
    optimizer.maximize(init_points=len(bounds.keys()), n_iter=search_iterations, acq=acq_func, kappa=kappa, xi=sigma_noise) # acq = "ucb", "ei", "poi"

    # Get end time
    end_time = time.time()
    print("\n" + time.strftime("[%H:%M:%S]", time.localtime(time.time())) + "[Bayesian Search Completed]")
    print("[Best Score: " + f'{optimizer.max["target"]}' + "]")

    # Save Results
    activation_value = int(math.ceil(optimizer.max['params']["activation_function"])) # (0, 1] = 1 relu, (1, 2] = 2 tanh, (2, 3] = 3 sigmoid

    if activation_value == 1:
        activation = "relu"
    elif activation_value == 2:
        activation = "tanh"
    elif activation_value == 3:
        activation = "sigmoid"

    stats = {
        "time_elapsed": end_time - start_time,
        "best_score": optimizer.max['target'],
        "best_parameters": {
            "batch_size": int(math.ceil(optimizer.max['params']["batch_size"])),
            "iterations": int(math.ceil(optimizer.max['params']["training_iterations"])),
            "activation": activation.upper(),
            "hidden_layers": int(math.ceil(optimizer.max['params']["hidden_layers"])),
            "neurons": int(math.ceil(optimizer.max['params']["neurons"])),
            "dropout_rate": optimizer.max['params']["dropout_rate"], 
            "learning_rate": optimizer.max['params']["learning_rate"], 
            "learning_decay": optimizer.max['params']["learning_decay"], 
            "weight_regularizer_l1": optimizer.max['params']["weight_regularizer_l1"], 
            "bias_regularizer_l1": optimizer.max['params']["bias_regularizer_l1"],
            "weight_regularizer_l2": optimizer.max['params']["weight_regularizer_l2"], 
            "bias_regularizer_l2": optimizer.max['params']["bias_regularizer_l2"]
        }
    }

    # Save Best Parameters
    save_parameters(stats, filename=param_file_path)

    # End Bayesian search