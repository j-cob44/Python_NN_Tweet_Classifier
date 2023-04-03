# Twitter_Data.py - Processes Tweet data for use in the Neural Network.
# Jacob Burton 2023

import json
import numpy as np
import re
import tweepy
import os

from __authsecrets__ import secret

# Dictionary for Neural Network Data Categories
nn_data_categories = {
    0: "Positive",
    1: "Negative"
}

# Twitter API v1.1 Authorization with keys from Secrets.py
auth = tweepy.OAuth2BearerHandler(secret.bearer_token)
api = tweepy.API(auth)

# Load Json
def load_json_file(path):
    with open(path, "r") as f:
        return json.load(f)

# Save Json
def save_json_file(tweet_data, path):
    with open(path, "w") as f:
        json.dump(tweet_data, f)

# Load Dataset
def load_tweet_data(path):
    # Instantiate empty lists
    X = [] # Samples
    y = [] # Labels (categories)

    # Load JSON
    data = load_json_file(path)

    for tweet in data:
        X.append(tweet[0])
        y.append(tweet[1])
    
    # Convert to numpy arrays
    return np.array(X), np.array(y).astype('uint8')

# Creating Dataset from Fashion MNIST
def create_tweet_datasets(path):
    # Loading the data
    print('Loading Tweet Data...')
    X, y = load_tweet_data(path)
    #X_test, y_test = load_mnist_fashion_data('test', path) # Validation Data

    return X, y #, X_test, y_test

# Preprocess Data
def preprocess_tweet_data(path, new_path=None):
    # Load Tweet Data
    tweets = []
    tweets = load_json_file(path)

    # Sort Tweets by Category (Positive, Negative)
    tweets.sort(key=lambda x: x[1])

    # Remove Duplicate Tweets
    for i in range(len(tweets) - 1):
        for j in range(i + 1, len(tweets)):
            if tweets[i][0] == tweets[j][0]:
                tweets.pop(j)
                break; 

    # Iterate through Tweets
    for i in range(len(tweets)):
        # Remove emoji's from tweets
        tweets[i][0] = tweets[i][0].encode('ascii', 'ignore').decode('ascii')

        # Use Regex to remove entire link from tweets
        tweets[i][0] = re.sub(r'http\S+', '', tweets[i][0])

        # Set all characters to lowercase
        tweets[i][0] = tweets[i][0].lower()

        # Remove all double spaces
        tweets[i][0] = tweets[i][0].replace("  ", " ")

        # If tweet is not 280 characters long, add spaces to end of tweet
        if len(tweets[i][0]) < 280:
            tweets[i][0] = tweets[i][0] + (" " * (280 - len(tweets[i][0])))

        # Turn string into list of characters
        tweets[i][0] = list(tweets[i][0])


    # Save Tweet Data
    if new_path != None:
        save_json_file(tweets, new_path)
    else:
        save_json_file(tweets, path)

# Analyze Data from Tweet Dataset
def analyze_dataset(path):
    # Load Tweet Data
    tweets = []
    tweets = load_json_file(path)

    # Analyze Data of Tweet Dataset
    total = len(tweets)
    postive_total = len([tweet for tweet in tweets if tweet[1] == 0])
    negative_total = len([tweet for tweet in tweets if tweet[1] == 1])

    print("Total Tweets: " + str(total))
    print("Positive Tweets: " + str(postive_total))
    print("Negative Tweets: " + str(negative_total))

    # Print length of longest string in dataset to ensure its 280 characters
    print("Longest Tweet: " + str(max([len(tweet[0]) for tweet in tweets])))

# Grab a tweet by ID
def grab_tweet(id):
    try:
        tweet = api.get_status(id, tweet_mode='extended')
    except:
        raise Exception("Error grabbing tweet.")
    
    # Skip Retweets
    if tweet.retweeted or 'RT @' in tweet.full_text:
        raise Exception("Tweet is a Retweet.")
    
    tweet = tweet.full_text.replace("\n", " ")

    return tweet

# Grab and process a single tweet from twitter
def grab_processed_tweet(id):
    # Get Tweet
    try:
        status = api.get_status(id, tweet_mode='extended')
    except:
        raise Exception("Error grabbing tweet.")

    # Skip Retweets
    if status.retweeted or 'RT @' in status.full_text:
        raise Exception("Tweet is a Retweet.")
    
    tweet = status.full_text.replace("\n", " ")

    # Remove emoji's from tweets
    tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    # Use Regex to remove entire link from tweets
    tweet = re.sub(r'http\S+', '', tweet)

    # Set all characters to lowercase
    tweet = tweet.lower()

    # Remove all double spaces
    tweet = tweet.replace("  ", " ")

    # If tweet is not 280 characters long, add spaces to end of tweet
    if len(tweet) < 280:
        tweet = tweet + (" " * (280 - len(tweet)))

    # Turn string into list of characters
    tweet = list(tweet)

    return tweet

# Final Processing of Tweet Array for Neural Network
def process_tweet_array_for_nn(tweets):
    # Convert ' ' to '`', '#' to '{', and '@' to '|' so when subtracted by 96, they will be 0, 27, and 28 respectively
    for i in range(len(tweets)):
        for j in range(len(tweets[i])):
            if tweets[i][j] == ' ':
                tweets[i][j] = '`' # ASCII value 96
            elif tweets[i][j] == '#':
                tweets[i][j] = '{' # ASCII value 123
            elif tweets[i][j] == '@':
                tweets[i][j] = '|' # ASCII value 124

    # Convert tweets into an array of an array of Integers
    tweets = np.array([[ord(c) for c in s] for s in tweets])

    # Subtract 96 from each letter to get a range of 0-28 (1-26 for letters, 27 for #, 28 for @, 0 for space)
    tweets = tweets - 96

    # Scale features between -1 and 1
    tweets = (tweets.astype(np.float32) - 14) / 14

    return tweets

# Final processing of a Single Tweet for Neural Network
def process_single_tweet_for_nn(tweet):
    # Convert ' ' to '`', '#' to '{', and '@' to '|' so when subtracted by 96, they will be 0, 27, and 28 respectively
    for i in range(len(tweet)):
        if tweet[i] == ' ':
            tweet[i] = '`' # ASCII value 96
        elif tweet[i] == '#':
            tweet[i] = '{' # ASCII value 123
        elif tweet[i] == '@':
            tweet[i] = '|' # ASCII value 124

    # Convert tweet into array of Integers
    tweet = np.array([ord(c) for c in tweet])

    # Subtract 96 from each letter to get a range of 0-28 (1-26 for letters, 27 for #, 28 for @, 0 for space)
    tweet = tweet - 96

    # Scale features between -1 and 1
    tweet = (tweet.astype(np.float32) - 14) / 14

    return tweet

# Add a tweet to the dataset
def add_tweet_by_id(tweet_id, category, dataset_path):
    # Get Tweets from Twitter API
    try:
        # Get Tweet
        status_text = grab_tweet(tweet_id)

        new_tweet = [status_text, int(category)]

        tweets = []
        if os.path.exists(dataset_path):
            tweets = load_json_file(dataset_path)

        tweets.append(new_tweet)

        save_json_file(tweets, dataset_path)

        return 1   
    except:
        print("Error submitting Tweet to submission set.")
        return 0

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

# Add a text based submission to a dataset
def add_text_submission(text, category, dataset_path):
    new_tweet = [text, int(category)]

    tweets = []
    if os.path.exists(dataset_path):
        tweets = load_json_file(dataset_path)

    tweets.append(new_tweet)

    save_json_file(tweets, dataset_path)

# Combine two datasets
def combine_tweet_datasets(dataset1_path, dataset2_path, new_dataset_path):
    # Initialize Datasets
    dataset1 = []
    dataset2 = []
    combined_dataset = []

    # Load Datasets
    try:
        dataset1 = load_json_file(dataset1_path)
    except:
        print("Error loading dataset 1.")
        return
    
    try:
        dataset2 = load_json_file(dataset2_path)
    except:
        print("Error loading dataset 2.")
        return

    # Combine Datasets
    combined_dataset = dataset1 + dataset2

    # Save Combined Dataset
    save_json_file(combined_dataset, new_dataset_path)