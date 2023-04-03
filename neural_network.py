# Neural_Network.py - Used for interacting with the Neural Network from the backend
# Jacob Burton 2023

from model_actions import *
from twitter_data import *

# Based on User Action
continue_actions = True
while continue_actions:
    user_action = input("What would you like to do? (t)rain, (r)etrain, (p)reprocess, (c)ombine datasets, (a)nalyze a processed dataset, (q)uit: ")

    # Train Model
    if user_action == "t":
        user_confirmation = input("Are you sure you want to train a new model? (y/n): ")
        if user_confirmation == "y":
            file_name = input("What would you like to name the model? ")
            model = create_model()
            X, y = create_dataset_and_process_data()
            model = train_model(model, X, y, iterations=10, batch_size=25, print_every=1)
            model.save('models/' + file_name + '.model')
        else:
            continue
    # Retrain Model
    elif user_action == "r":
        file_name = input("What is the file name of the model you would like to retrain? ")
        try:
            model = Model.load('models/' + file_name + '.model')
            X, y = create_dataset_and_process_data()
            model = train_model(model, X, y, iterations=10, batch_size=25, print_every=1)
            model.save('models/' + file_name + '.model')
        except:
            print("Model not found!")
            continue
    # Preprocess Data
    elif user_action == "p":
        file_name = input("What is the file name of the data you would like to preprocess? ")
        preprocess_tweet_data("tweet_data/" + file_name + ".json", "tweet_data/" + file_name + "_processed.json")
        analyze_dataset("tweet_data/" + file_name + "_processed.json")
    # Combine Datasets
    elif user_action == "c":
        file_name_1 = input("What is the file name of the first dataset you would like to combine? ")
        file_name_2 = input("What is the file name of the second dataset you would like to combine? ")
        new_file_name = input("What would you like to name the combined dataset? ")
        combine_tweet_datasets("tweet_data/" + file_name_1 + ".json", "tweet_data/" + file_name_2 + ".json", "tweet_data/" + new_file_name + "_combined.json")
    # Analyze Processed Dataset
    elif user_action == "a":
        file_name = input("What is the file name of the dataset you would like to analyze? ")
        analyze_dataset("tweet_data/" + file_name + ".json")
    # Quit
    elif user_action == "q":
        continue_actions = False
    # Invalid Input
    else:
        print("Invalid Input!")
        
    
#########################################################