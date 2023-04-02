# Model_Actions.py - Contains all functions for interacting with the Neural Network
# Jacob Burton 2023

import numpy as np

from model import *
from twitter_data import *

# Creation of Dataset and Final Data Manipulation
def create_dataset_and_process_data():
    # Preprocessing of Data
    preprocess_tweet_data("tweet_data/training/nn_tweet_data.json", "tweet_data/training/nn_tweet_data_processed.json")

    # X stands for Actual Training Dataset
    X, y = create_tweet_datasets('tweet_data/training/nn_tweet_data_processed.json')

    # Randomly shuffle the data
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    X = process_tweet_array_for_nn(X)

    return X, y

##########################################################
# Creation of Neural Network Model
def create_model():
    model = Model()

    # Adding Layers to Model
    model.add(Layer_Dense(280, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 2))
    model.add(Activation_Softmax())

    # Set Model's Loss, Optimizer, and Accuracy Functions
    model.set(
        loss = Loss_CategoricalCrossEntropy(),
        optimizer = Optimizer_Adam(decay=1e-3),
        accuracy = Accuracy_Categorical()
    )

    # Finalize the Model
    model.finalize()

    return model

##########################################################
# Train/Retrain model
def train_model(model, X, y, iterations=10, batch_size=25, print_every=1):
    # Train the Model
    model.train(X,y, iterations=10, batch_size=25, print_every=1)#, validation_data=(X_test, y_test))

    return model

##########################################################
# Save model
def save_model(model, path):
    model.save(path)

##########################################################
# Load model
def load_model(path):
    model = Model.load(path)
    return model

##########################################################
# Evaluate on a single tweet
def evaluate_tweet(model, tweet_id):
    try:
        tweet = grab_processed_tweet(tweet_id)
    except Exception as e:
        raise e

    # Process Tweet For Neural Network Input
    tweet = process_single_tweet_for_nn(tweet)

    # Make Prediction on Tweet
    confidences = model.predict(tweet)

    # Get Prediction from Confidence Level
    predictions = model.output_layer_activation.predictions(confidences)

    # # Print Tweet
    #raw_tweet = grab_tweet(tweet_id)
    #tweet_text = raw_tweet.full_text.replace("\n", " ")
    #print("\n" + tweet_text + "\n")

    # Get Category and Print
    for prediction in predictions:
        highest_confidence_as_percent = np.max(confidences) * 100
        #print("Network is", f'{highest_confidence_as_percent:.3f}%', "confident this tweet is", nn_data_categories[prediction])
    
    return highest_confidence_as_percent, nn_data_categories[prediction]

