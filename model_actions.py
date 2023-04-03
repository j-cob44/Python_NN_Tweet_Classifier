# Model_Actions.py - Contains all functions for interacting with the Neural Network
# Jacob Burton 2023

import numpy as np

from model import *
from twitter_data import *

# Creation of Dataset and Final Data Manipulation
def create_dataset_and_process_data(path):
    # Preprocessing of Data
    preprocess_tweet_data("tweet_data/" + path + ".json", "tweet_data/" + path + "_processed.json")

    # X stands for Actual Training Dataset
    X, y = create_tweet_datasets('tweet_data/'+ path + '_processed.json')

    # Randomly shuffle the data
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    X = process_tweet_array_for_nn(X)

    return X, y

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

# Train/Retrain model
def train_model(model, X, y, iterations=10, batch_size=25, print_every=1):
    # Train the Model
    model.train(X,y, iterations=iterations, batch_size=batch_size, print_every=print_every)#, validation_data=(X_test, y_test))

    return model

# Save model
def save_model(model, path):
    model.save(path)

# Load model
def load_model(path):
    model = Model.load(path)
    return model

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

# Process Text for Neural Network Input - Input as string, output as list of characters
def process_text_for_nn(text):
    # Process text For Neural Network Input
    text = text.replace("\n", " ")

    # Remove emoji's from tweets
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Use Regex to remove entire link from tweets
    text = re.sub(r'http\S+', '', text)

    # Set all characters to lowercase
    text = text.lower()

    # Remove all double spaces
    text = text.replace("  ", " ")

    # If tweet is not 280 characters long, add spaces to end of tweet
    if len(text) < 280:
        text = text + (" " * (280 - len(text)))

    # Turn string into list of characters
    text = list(text)

    return text

# Evaluate text in the Neural Network
def evaluate_text(model, text):
    # Process Text For Neural Network Input
    text = process_text_for_nn(text)

    # Process "Tweet" for Neural Network Input
    text = process_single_tweet_for_nn(text)

    # Make Prediction on Text
    confidences = model.predict(text)

    # Get Prediction from Confidence Level
    predictions = model.output_layer_activation.predictions(confidences)

    # Get Category and Print
    for prediction in predictions:
        highest_confidence_as_percent = np.max(confidences) * 100
        #print("Network is", f'{highest_confidence_as_percent:.3f}%', "confident this tweet is", nn_data_categories[prediction])
    
    return highest_confidence_as_percent, nn_data_categories[prediction]

# Save Model data to file
def save_model_data(model, path):
    data = {
        "loss_function": model.loss.__class__.__name__,
        "optimizer_function": model.optimizer.__class__.__name__,
        "accuracy_function": model.accuracy.__class__.__name__,
        "calculated_accuracy": model.accuracy.calculate_accumulated()
    }

    with open(path, "w") as f:
        json.dump(data, f)