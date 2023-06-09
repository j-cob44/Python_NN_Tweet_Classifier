# Model_Actions.py - Contains all functions for interacting with the Neural Network
# Jacob Burton 2023

import numpy as np
import json
import matplotlib.pyplot as plt
import datetime

from model import *
from tweet_data import *

# Creation of Neural Network Model
def create_model(
        hidden_layers=2, 
        neurons=64, 
        dropout=0.25, 
        learning_rate=0.1, 
        learning_decay=1e-3, 
        weight_regularizer_l1=0, 
        weight_regularizer_l2=0.01,
        bias_regularizer_l1=0,
        bias_regularizer_l2=0.01,
        activation="relu"):
    
    # Initialize Model
    model = Model()

    inputs = 280 * possible_chars # 280 (length of tweet) * possible characters

    # Adding Layers to Model
    for i in range(hidden_layers):
        if i == 0:
            model.add(Layer_Dense(inputs, neurons,
                weight_regularizer_l1=weight_regularizer_l1,
                bias_regularizer_l1=bias_regularizer_l1,
                weight_regularizer_l2=weight_regularizer_l2, 
                bias_regularizer_l2=bias_regularizer_l2
            ))
        else:
            model.add(Layer_Dense(neurons, neurons,
                weight_regularizer_l1=weight_regularizer_l1,
                bias_regularizer_l1=bias_regularizer_l1,
                weight_regularizer_l2=weight_regularizer_l2, 
                bias_regularizer_l2=bias_regularizer_l2
            ))
        
        # Select Activation function for hidden layers
        if activation == "relu":
            model.add(Activation_ReLU())
        elif activation == "tanh":
            model.add(Activation_Tanh())
        elif activation == "sigmoid":
            model.add(Activation_Sigmoid())
        elif activation == "leakyrelu":
            model.add(Activation_LeakyReLU())
        else:
            raise Exception("Activation Function not recognized")
        
        model.add(Layer_Dropout(dropout))
        
    # Output Layer
    model.add(Layer_Dense(neurons, 1))
    model.add(Activation_Sigmoid())

    # Set Model's Loss, Optimizer, and Accuracy Functions
    model.set(
        loss = Loss_BinaryCrossEntropy(),
        optimizer = Optimizer_Adam(
            learning_rate=learning_rate, 
            decay=learning_decay),
        accuracy = Accuracy_Binary()
    )

    # Finalize the Model
    model.finalize()

    return model

# Train/Retrain model
def train_model(model, X, y, X_val=None, y_val=None, iterations=10, batch_size=64, print_every=100):
    # Get time
    start_time = time.time()

    if X_val is not None and y_val is not None:
        # Train and Validate the Model
        training_data = model.train(X,y, iterations=iterations, batch_size=batch_size, print_every=print_every, validation_data=(X_val, y_val))
    else:
        # Only Train the Model
        training_data = model.train(X,y, iterations=iterations, batch_size=batch_size, print_every=print_every)

    # Get time
    end_time = time.time()

    analysis_data = {
        "time_to_train": end_time - start_time,
        "loss_function": model.loss.__class__.__name__,
        "optimizer": model.optimizer.__class__.__name__,
        "accuracy_function": model.accuracy.__class__.__name__,
        "training_data": training_data
    }

    return model, analysis_data

# Save model to file
def save_model(model, path):
    model.save(path)

# Load model from file
def load_model(path):
    model = Model.load(path)
    return model

# Save training data to file
def save_training_data(training_data, path):
    with open(path, "w") as f:
        json.dump(training_data, f, indent=4)

# Analyze model training data
def analyze_training_data(path):
    # Get training data from json file
    with open(path, "r") as f:
        analysis_data = json.load(f)

    training_data = analysis_data.get("training_data")

    # Get training time
    minutes = (analysis_data.get("time_to_train")/60)
    seconds = (minutes - int(minutes)) * 60

    # Get Analysis Data
    print("Minutes to Train: " + f'{int(minutes)}' + ":" + f'{int(seconds)}')
    print("") # New Line
    print("Loss Function: " + analysis_data.get("loss_function"))
    print("Optimizer: " + analysis_data.get("optimizer"))
    print("Accuracy Function: " + analysis_data.get("accuracy_function"))

    print("") # New Line
    print("Training Accuracy: " + f'{(training_data.get("training_accuracy") * 100):.2f}' + "%")
    if training_data.get("validation_accuracy") != 0:
        print("Validation Accuracy: " + f'{(training_data.get("validation_accuracy") * 100):.2f}' + "%")
    if training_data.get("validation_loss") != 0:
        print("Validation Loss: " + f'{(training_data.get("validation_loss")):.8f}')

    # Draw Plots
    y1 = training_data.get("loss_history")
    plt.subplot(3,1,1)
    plt.title("Loss")
    plt.plot(range(len(y1)), y1)

    y2 = training_data.get("accuracy_history")
    plt.subplot(3,1,2)
    plt.title("Accuracy")
    plt.ylim(0, 1)
    plt.plot(range(len(y2)), y2)

    y3 = training_data.get("learning_rate_history")
    plt.subplot(3,1,3)
    plt.title("Learning Rate")
    plt.plot(range(len(y3)), y3)

    # Add space between subplots
    plt.subplots_adjust(hspace=1)

    # Display
    plt.show()

#!!! Twitter API Depreciated !!! 
# Evaluate on a single tweet
def evaluate_tweet(model, tweet_id):
    pass
    # try:
    #     tweet = grab_processed_tweet(tweet_id)
    # except Exception as e:
    #     raise e

    # # Process Tweet For Neural Network Input
    # tweet = process_single_tweet_for_nn(tweet)

    # # Make Prediction on Tweet
    # confidences = model.predict(tweet)

    # # Get Prediction from Confidence Level
    # predictions = model.output_layer_activation.predictions(confidences)

    # # # Print Tweet
    # #raw_tweet = grab_tweet(tweet_id)
    # #tweet_text = raw_tweet.full_text.replace("\n", " ")
    # #print("\n" + tweet_text + "\n")

    # Get Category and Print
    # highest_confidence_as_percent = (np.abs(confidences[0][0] - 0.5) + 0.5) * 100 # Scale value 
    # prediction = predictions[0][0]

    # return highest_confidence_as_percent, nn_data_categories[prediction]

# Process Text for Neural Network Input - Input as string, output as one-hot encoded vector
def process_text_for_nn(text):
    # Lowercase all letters
    text = text.lower()

    # Remove stop words
    text = remove_stopwords(text)

    # Replace Apostrophes
    text = text.replace('‘', "'").replace('’', "'")

    # Remove all characters that are not in the character dictionary
    for char in text:
        if char not in char_dict:
            text = text.replace(char, '')
            pass
    
    # Remove '\n' characters
    text = text.replace('\n', ' ')

    # Make sure all tweets are 280 characters long
    if len(text) < max_tweet_length:
        # Add spaces to the end of the tweet to make it 280 characters long
        text = text.ljust(max_tweet_length, ' ')

    # Character-level One-Hot Encode Tweet text
    one_hot_text = one_hot_encode_single_tweet(text)

    return one_hot_text

# Evaluate text in the Neural Network
def evaluate_text(model, text):
    # Process Text For Neural Network Input
    one_hot_vector = process_text_for_nn(text)

    # Reshape Input
    data_input = one_hot_vector.reshape(one_hot_vector.shape[0] * one_hot_vector.shape[1])

    # Make Prediction on Input
    confidences = model.predict(data_input)

    # Get Prediction from Confidence Level
    predictions = model.output_layer_activation.predictions(confidences)

    # Get Category and Print
    highest_confidence_as_percent = (np.abs(confidences[0][0] - 0.5) + 0.5) * 100 # Scale value 
    prediction = predictions[0][0]

    return highest_confidence_as_percent, nn_data_categories[prediction]

#!!! Twitter API Depreciated !!!
# Add a tweet to the database for further training
def add_tweet_by_id(tweet, category, submission_path):
    pass

# Add text submission to the database for further training
def add_text_submission(text, category, submission_path):
    # Get current time
    current_time = datetime.datetime.now()

    # Create submission
    submission = {
        "text": text,
        "category": category,
        "time": current_time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Add submission to database
    with open(submission_path, "r") as f:
        submissions = json.load(f)
    
    submissions.append(submission)

    with open(submission_path, "w") as f:
        json.dump(submissions, f, indent=4)