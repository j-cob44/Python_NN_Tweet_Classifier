# Tweet_Data.py - Processes Tweet data for use in the Neural Network.
# Jacob Burton 2023

import tweepy
import numpy as np
import pandas as pd

from __authsecrets__ import secret

# Dictionary for Neural Network Data Categories
nn_data_categories = {
    1: "Positive",
    0: "Negative"
}

# Character Dictionary - Used for One-Hot Encoding
char_dict = [' ', 
#    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    '!', '?', '$', "'", ',', '.', '0', '1', '2', '3', '4', '5', '6', '7', '9', ':', '@']
possible_chars = len(char_dict)

# Stop Words Dictionary
stop_words = [ "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", 
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"]

# Maximum Tweet Length
max_tweet_length = 280

# Twitter API v1.1 Authorization with keys from Secrets.py
auth = tweepy.OAuth2BearerHandler(secret.bearer_token)
api = tweepy.API(auth)

# Load CSV File | Input: path to csv file, Output: pandas dataframe
def load_csv_file(path):
    reader = pd.read_csv(path, encoding='ISO-8859-1')
    reader.columns = ['id', 'topic', 'label', 'text']
    
    return reader

# Process Dataset | Input: path to csv file, Output: pandas dataframe
def process_dataset_from_csv(path):
    # Load CSV File
    dataset = load_csv_file(path)

    # Remove id and topic columns
    dataset = dataset.drop(['id', 'topic'], axis=1)

    # Remove label column that are 'Irrelevant'
    dataset = dataset[dataset.label != 'Irrelevant']

    # Remove label comumn that are 'Neutral'
    dataset = dataset[dataset.label != 'Neutral']

    # Convert labels to 1 for positive and -1 for negative and 0 for neutral
    dataset['label'] = dataset['label'].replace(['Positive', 'Negative'], [1, 0])

    # Datset remove empty rows
    dataset = dataset.dropna()

    # Ensure dataset is balanced
    n_samples = min(len(dataset[dataset['label'] == 1]), len(dataset[dataset['label'] == 0]))

    # Split dataset into positives and negatives
    positives = dataset[dataset['label'] == 1]
    negatives = dataset[dataset['label'] == 0]

    # Randomly sample n_samples from the bigger class
    if( len(dataset[dataset['label'] == 1]) > len(dataset[dataset['label'] == 0]) ):
        # Positives class is bigger
        positives = positives.sample(n=n_samples, replace=False) 
        dataset = pd.concat([positives, negatives]) # Rejoin the datasets
    elif ( len(dataset[dataset['label'] == 1]) < len(dataset[dataset['label'] == 0]) ):
        # Negatives class is bigger
        negatives = negatives.sample(n=n_samples, replace=False)
        dataset = pd.concat([positives, negatives]) # Rejoin the datasets
    else:
        # balanced
        pass

    # Change data into lowercase
    dataset['text'] = dataset['text'].str.lower()

    return dataset

# Remove Stop Words | Input: string, Output: string
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Process Tweet Data | Input: numpy array of tweets, Output: numpy array of processed tweets
def process_tweet_data(tweet_dataset):
    # For each tweet in the dataset
    for i in range(len(tweet_dataset)) :
        # Remove stop words
        tweet_dataset[i] = remove_stopwords(tweet_dataset[i])

        # Replace Apostrophes
        tweet_dataset[i] = tweet_dataset[i].replace('‘', "'").replace('’', "'")

        # Remove all characters that are not in the character dictionary
        for char in tweet_dataset[i]:
            if char not in char_dict:
                tweet_dataset[i] = tweet_dataset[i].replace(char, '')
                pass
        
        # Remove '\n' characters
        tweet_dataset[i] = tweet_dataset[i].replace('\n', ' ')

        # Make sure all tweets are 280 characters long
        if len(tweet_dataset[i]) < max_tweet_length:
            # Add spaces to the end of the tweet to make it 280 characters long
            tweet_dataset[i] = tweet_dataset[i].ljust(max_tweet_length, ' ')

    # Character-level One-Hot Encode all tweets
    tweet_dataset = one_hot_encode_tweet_dataset(tweet_dataset)

    return tweet_dataset

# One-hot encode all tweets in the dataset | Input: numpy array of tweets, Output: numpy array of one-hot encoded tweets
def one_hot_encode_tweet_dataset(tweets):
    # Create Zero'd array
    one_hot = np.zeros((len(tweets), max_tweet_length, len(char_dict)))

    # Iterate over each string, then each character in that string, setting the appropriate index to 1
    for i in range(len(tweets)):
        for j in range(280):
            one_hot[i, j, char_dict.index(tweets[i][j])] = 1
    return one_hot

# Load Dataset | Input: path to dataset, Output: numpy array X, numpy array y
def load_tweet_data(path):
    # Instantiate empty lists
    X = [] # Samples
    y = [] # Labels (categories)

    # Create Dataset
    dataset = process_dataset_from_csv(path)  

    # Separate samples and labels
    X = dataset['text'] # Tweets
    X = process_tweet_data(X.to_numpy())

    y = dataset['label']
    y = y.to_numpy()
    y = y.reshape(y.shape[0], 1) # Reshape y into (n, 1) 2D array

    return X, y

# Creating the Dataset of tweets
def create_tweet_datasets(trainingPath, validationPath=None):
    # Loading the data
    print('Loading Training Data...')
    X, y = load_tweet_data(trainingPath)

    # Shuffle the data
    print('Shuffling Training Data...')
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    if validationPath is not None:
        print('Loading Validation Data...')
        X_val, y_val = load_tweet_data(validationPath)

        print('Shuffling Validation Data...')
        keys = np.array(range(X_val.shape[0]))
        np.random.shuffle(keys)
        X_val = X_val[keys]
        y_val = y_val[keys]

        return X, y, X_val, y_val
    else:
        return X, y, None, None

# TODO: Implement Following Functions

# Analyze Data from Tweet Dataset
# def analyze_dataset(path):
#     # Load Tweet Data
#     tweets = []
#     tweets = load_json_file(path)

#     # Analyze Data of Tweet Dataset
#     total = len(tweets)
#     postive_total = len([tweet for tweet in tweets if tweet[1] == 0])
#     negative_total = len([tweet for tweet in tweets if tweet[1] == 1])

#     print("Total Tweets: " + str(total))
#     print("Positive Tweets: " + str(postive_total))
#     print("Negative Tweets: " + str(negative_total))

#     # Print length of longest string in dataset to ensure its 280 characters
#     print("Longest Tweet: " + str(max([len(tweet[0]) for tweet in tweets])))

# Grab a tweet by ID
def grab_tweet(id):
    # try:
    #     tweet = api.get_status(id, tweet_mode='extended')
    # except:
    #     raise Exception("Error grabbing tweet.")
    
    # # Skip Retweets
    # if tweet.retweeted or 'RT @' in tweet.full_text:
    #     raise Exception("Tweet is a Retweet.")
    
    # tweet = tweet.full_text.replace("\n", " ")

    # return tweet
    pass

# Grab and process a single tweet from twitter
def grab_processed_tweet(id):
    # # Get Tweet
    # try:
    #     status = api.get_status(id, tweet_mode='extended')
    # except:
    #     raise Exception("Error grabbing tweet.")

    # # Skip Retweets
    # if status.retweeted or 'RT @' in status.full_text:
    #     raise Exception("Tweet is a Retweet.")
    
    # tweet = status.full_text.replace("\n", " ")

    # # Remove emoji's from tweets
    # tweet = tweet.encode('ascii', 'ignore').decode('ascii')

    # # Use Regex to remove entire link from tweets
    # tweet = re.sub(r'http\S+', '', tweet)

    # # Set all characters to lowercase
    # tweet = tweet.lower()

    # # Remove all double spaces
    # tweet = tweet.replace("  ", " ")

    # # If tweet is not 280 characters long, add spaces to end of tweet
    # if len(tweet) < 280:
    #     tweet = tweet + (" " * (280 - len(tweet)))

    # # Turn string into list of characters
    # tweet = list(tweet)

    # return tweet
    pass

# Final Processing of Tweet Array for Neural Network
def process_tweet_array_for_nn(tweets):
    # # Convert ' ' to '`', '#' to '{', and '@' to '|' so when subtracted by 96, they will be 0, 27, and 28 respectively
    # for i in range(len(tweets)):
    #     for j in range(len(tweets[i])):
    #         if tweets[i][j] == ' ':
    #             tweets[i][j] = '`' # ASCII value 96
    #         elif tweets[i][j] == '#':
    #             tweets[i][j] = '{' # ASCII value 123
    #         elif tweets[i][j] == '@':
    #             tweets[i][j] = '|' # ASCII value 124

    # # Convert tweets into an array of an array of Integers
    # tweets = np.array([[ord(c) for c in s] for s in tweets])

    # # Subtract 96 from each letter to get a range of 0-28 (1-26 for letters, 27 for #, 28 for @, 0 for space)
    # tweets = tweets - 96

    # # Scale features between -1 and 1
    # tweets = (tweets.astype(np.float32) - 14) / 14

    # return tweets
    pass

# Final processing of a Single Tweet for Neural Network
def process_single_tweet_for_nn(tweet):
    # # Convert ' ' to '`', '#' to '{', and '@' to '|' so when subtracted by 96, they will be 0, 27, and 28 respectively
    # for i in range(len(tweet)):
    #     if tweet[i] == ' ':
    #         tweet[i] = '`' # ASCII value 96
    #     elif tweet[i] == '#':
    #         tweet[i] = '{' # ASCII value 123
    #     elif tweet[i] == '@':
    #         tweet[i] = '|' # ASCII value 124

    # # Convert tweet into array of Integers
    # tweet = np.array([ord(c) for c in tweet])

    # # Subtract 96 from each letter to get a range of 0-28 (1-26 for letters, 27 for #, 28 for @, 0 for space)
    # tweet = tweet - 96

    # # Scale features between -1 and 1
    # tweet = (tweet.astype(np.float32) - 14) / 14

    # return tweet
    pass

# Add a tweet to the dataset
# def add_tweet_by_id(tweet_id, category, dataset_path):
#     # Get Tweets from Twitter API
#     try:
#         # Get Tweet
#         status_text = grab_tweet(tweet_id)

#         new_tweet = [status_text, int(category)]

#         tweets = []
#         if os.path.exists(dataset_path):
#             tweets = load_json_file(dataset_path)

#         tweets.append(new_tweet)

#         save_json_file(tweets, dataset_path)

#         return 1   
#     except:
#         print("Error submitting Tweet to submission set.")
#         return 0

# Process Text for Neural Network Input - Input as string, output as list of characters
def process_text_for_nn(text):
    # # Process text For Neural Network Input
    # text = text.replace("\n", " ")

    # # Remove emoji's from tweets
    # text = text.encode('ascii', 'ignore').decode('ascii')

    # # Use Regex to remove entire link from tweets
    # text = re.sub(r'http\S+', '', text)

    # # Set all characters to lowercase
    # text = text.lower()

    # # Remove all double spaces
    # text = text.replace("  ", " ")

    # # If tweet is not 280 characters long, add spaces to end of tweet
    # if len(text) < 280:
    #     text = text + (" " * (280 - len(text)))

    # # Turn string into list of characters
    # text = list(text)

    # return text
    pass

# Add a text based submission to a dataset
# def add_text_submission(text, category, dataset_path):
#     new_tweet = [text, int(category)]

#     tweets = []
#     if os.path.exists(dataset_path):
#         tweets = load_json_file(dataset_path)

#     tweets.append(new_tweet)

#     save_json_file(tweets, dataset_path)
