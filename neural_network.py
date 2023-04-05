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

            # Create the model
            model = create_model()

            # Check if X is 3D
            if len(X.shape) == 3:
                # Make X 2D
                X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

            # Training Parameters
            epochs = 10
            batch_size = 64
            print_every = 20

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
        print(""); # New Line only

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

            print("\nWhere would you like to save the results to?")
            save_path = input("parameter_data/")
            parameter_save_path = "parameter_data/" + save_path

            print(""); # New Line for viewing pleasure
            perform_grid_search(parameter_save_path, X, y, X_val, y_val,
                hidden_layers = [1, 2],  # Number of hidden layers
                neurons = [64, 128, 256],  # Number of neurons in each hidden layer
                dropouts = [0.1, 0.2, 0.25, 0.3],  # Dropout rate
                learning_rates = [0.001, 0.01, 0.0001, 0.02],  # Learning rate
                learning_decays = [1e-4, 5e-5, 1e-3, 1e-5, 5e-7],  # Learning rate decay
                #weight_regularizers_l1 = [0.001, 0.01, 0.1],  # L1 weight regularization
                #bias_regularizers_l1 = [0.001, 0.01, 0.1],  # L1 bias regularization
                weight_regularizers_l2 = [1e-4, 5e-5, 1e-5],  # L2 weight regularization
                bias_regularizers_l2 = [1e-4, 5e-5, 1e-5])  # L2 bias regularization
        
    # Quit
    elif user_action == "q":
        continue_actions = False
    # Invalid Input
    else:
        print("Invalid Input!")
        
    
#########################################################
