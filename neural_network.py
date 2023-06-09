# Neural_Network.py - Used for interacting with the Neural Network from the backend
# Jacob Burton 2023

from model_actions import *
from tweet_data import *
from parameter_tuning import *

# Based on User Action
continue_actions = True
while continue_actions:
    user_action = input("What would you like to do? (t)rain, (r)etrain, (a)nalyze a model, (h)yperparameter tune, (q)uit: ")

    # Train Model
    if user_action == "t":
        user_confirmation = input("Are you sure you want to train a new model? (y/n): ")
        if user_confirmation == "y":
            file_name = input("What would you like to name the model? ")
            print("What is the file name of the training dataset you would like to train the model on? ")
            training_path = input("tweet_data/")
            print("What is the file name of the validation dataset you would like to validate the model with? ")
            validation_path = input("tweet_data/")
            print("");

            # Create the datasets
            full_t_path = "tweet_data/" + training_path
            if validation_path != "":
                full_v_path = "tweet_data/" + validation_path
            else:
                full_v_path = None
            X, y, X_val, y_val = create_tweet_datasets(full_t_path, full_v_path)

            # Check if X is 3D
            if len(X.shape) == 3:
                # Make X 2D
                X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

            # Create the model
            model = create_model(
                hidden_layers=2, 
                neurons=291, 
                dropout=0.12678879420467504, 
                learning_rate=0.009893731095534928, 
                learning_decay=0.009807068792481532, 
                weight_regularizer_l1=0, 
                weight_regularizer_l2=0.00028695964014064014, 
                bias_regularizer_l1=0, 
                bias_regularizer_l2=0.0007938889471017285, 
                activation="relu" # relu, tanh, sigmoid, leakyrelu
            )

            # Training Parameters
            epochs = 38 # training iterations
            batch_size = 66
            print_every = 25

            # Check if validation data is provided
            if X_val is not None and y_val is not None:
                # Check if X_val is 3D
                if len(X_val.shape) == 3:
                    # Make X_val 2D
                    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])

                # Train and Validate the Model. Returns the model and training data object
                model, training_data = train_model(model, X, y, X_val, y_val, iterations=epochs, batch_size=batch_size, print_every=print_every)
            else:
                # Only Train the Model. Returns the model and training data object
                model, training_data = train_model(model, X, y, iterations=epochs, batch_size=batch_size, print_every=print_every)
            
            model.save('models/' + file_name + '.model')
            save_training_data(training_data, 'models/' + file_name + '_info.json')
        else:
            continue
    # Retrain Model
    elif user_action == "r":
        file_name = input("What is the file name of the model you would like to retrain? ")
        data_set = input("What is the file name of the dataset you would like to train the model on? ")
        try:
            model = Model.load('models/' + file_name + '.model') 
            X, y = create_tweet_datasets(data_set)

            model, training_data = train_model(model, X, y, iterations=10, batch_size=25, print_every=1)

            model.save('models/' + file_name + '.model')
            save_training_data(model, 'models/' + file_name + '_info.json')
        except:
            print("Model not found!")
            continue
    # Analyze Model
    elif user_action == "a":
        print("What is the file name of the dataset you would like to analyze?")
        file_name = input("models/")
        print(""); # New Line

        # Analyze and display stats
        analyze_training_data("models/" + file_name)

    # Automatic Hyperparameter Tuning
    elif user_action == "h":
        print("What is the file name of the training dataset you would like to train the model on? ")
        training_path = input("tweet_data/")
        print("What is the file name of the validation dataset you would like to validate the model with? ")
        validation_path = input("tweet_data/")
        print(""); # New Line 

        # Create the datasets
        full_t_path = "tweet_data/" + training_path
        if validation_path == "":
            print("No validation dataset provided. Validation dataset is required.")
        else:
            full_v_path = "tweet_data/" + validation_path
            X, y, X_val, y_val = create_tweet_datasets(full_t_path, full_v_path)
            
            # Reshape 3D data to 2D
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            if len(X_val.shape) == 3:
                X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])

            # Get User Input
            search_type = input("What search would you like to do? (g)rid search, (r)andom search, (b)ayesian optimization: ")

            print("\nWhere would you like to save the results to?")
            save_path = input("parameter_data/")
            parameter_save_path = "parameter_data/" + save_path
            print(""); # New Line for viewing pleasure

            if(search_type == "g"):
                # Perform Grid Search
                perform_grid_search(parameter_save_path, X, y, X_val, y_val,
                    hidden_layers = [1, 2], # Number of hidden layers
                    neurons = [64, 128, 256], # Number of neurons in each hidden layer
                    dropouts = [0.1, 0.2, 0.25, 0.3], # Dropout rate
                    learning_rates = [0.001, 0.01, 0.0001, 0.02], # Learning rate
                    learning_decays = [1e-4, 5e-5, 1e-3, 1e-5, 5e-7], # Learning rate decay
                    #weight_regularizers_l1 = [0.001, 0.01, 0.1], # L1 weight regularization # broken?
                    #bias_regularizers_l1 = [0.001, 0.01, 0.1], # L1 bias regularization # broken?
                    weight_regularizers_l2 = [1e-4, 5e-5, 1e-5], # L2 weight regularization
                    bias_regularizers_l2 = [1e-4, 5e-5, 1e-5] # L2 bias regularization
                )
            elif(search_type == "r"):
                # Perform Random Search
                random_search(parameter_save_path, X, y, X_val, y_val, search_iterations=50,
                    training_iterations = [10, 20, 30], # Number of training iterations
                    batch_size = [32, 64, 128, 256], # Training Batch size
                    hidden_layers = [1, 2, 3, 4], # Number of hidden layers
                    neurons = [32, 64, 128, 256], # Number of neurons in each hidden layer
                    dropouts = [0.1, 0.2, 0.25, 0.3, 0.5], # Dropout rate
                    learning_rates = [0.001, 0.01, 0.0001, 0.005, 0.00005], # Learning rate
                    learning_decays = [1e-4, 5e-5, 1e-3, 1e-5], # Learning rate decay
                    weight_regularizers_l1 = [0, 0.001, 0.01], # L1 weight regularization
                    bias_regularizers_l1 = [0, 0.001, 0.01], # L1 bias regularization
                    weight_regularizers_l2 = [1e-4, 5e-5, 1e-5], # L2 weight regularization
                    bias_regularizers_l2 = [1e-4, 5e-5, 1e-5] # L2 bias regularization
                )
            elif(search_type == "b"):
                print(""); # New Line 
                # Perform Bayesian Optimization Search
                bayesian_search(parameter_save_path, X, y, X_val, y_val, 
                    search_iterations=100, 
                    acq_func='poi', # Acquisition function = 'ucb', 'ei', or 'poi'
                    kappa=2.576, 
                    sigma_noise=1e-6,
                    target="accuracy", # Target = max 'accuracy', min 'loss', or 'both'
                    param_space = {
                        # Model Parameters
                        "weight_regularizer_l1": (0, 0),
                        "bias_regularizer_l1": (0, 0),
                        "weight_regularizer_l2": (0, 1e-3),
                        "bias_regularizer_l2": (0, 1e-3),
                        "learning_decay": (1e-5, 0.01),
                        "learning_rate": (0.005, 0.025),
                        "dropout_rate": (0.1, 0.3),
                        "neurons": (int(149 + 1e-4), int(300)), # [(149, 150] = 150, 300]
                        "hidden_layers": (int(1 + 1e-4), int(3)), # (1, 2] = 2, (2, 3] = 3
                        "activation_function": (int(1 + 1e-4), int(2)), # (0, 1] = relu, (1, 2] = tanh, (2, 3] = sigmoid
                        # Training Parameters
                        "batch_size": (int(39 + 1e-4), int(100)), # (31, 32] = 32
                        "training_iterations": (int(19 + 1e-4), int(50)) # (7, 8] = 8, (8, 9] = 9, (31, 32] = 32
                    }
                )
        
    # Quit
    elif user_action == "q":
        continue_actions = False
    # Invalid Input
    else:
        print("Invalid Input!")
        
#########################################################
