# Twitter_Data_By_ID.py - Grabs a specific Tweet by ID for Analysis
# Jacob Burton 2023

import tweepy
import json
import numpy as np
import os
import sys

from __authsecrets__ import secret
from twitter_data import nn_data_categories

# Data path (Change when creating different datasets)
data_path = "tweet_data/training/nn_tweet_data.json"

# Twitter API v1.1 Authorization with keys from Secrets.py
auth = tweepy.OAuth2BearerHandler(secret.bearer_token)
api = tweepy.API(auth)

# Load Old Tweet Data
tweets = []
with open(data_path, "r") as f:
    tweets = json.load(f)

# Print Current Collection Information
print("Postive Tweets Total: " + str(len([tweet for tweet in tweets if tweet[1] == 0])))
print("Negative Tweets Total: " + str(len([tweet for tweet in tweets if tweet[1] == 1])))

# Create List of Tweets
new_tweets = []

# Loop for gathering Data
continue_loop = True
while continue_loop:
    user_input = input("Enter Tweet ID (0 to Quit): ")

    # Check User Input
    if user_input == "0":
        print("Stopping.")
        continue_loop = False
        break;

    # Get Tweets from Twitter API
    try:
        # Get Tweet
        status = api.get_status(user_input, tweet_mode='extended')

        # Skip Retweets
        if status.retweeted or 'RT @' in status.full_text:
            print("Tweet is a Retweet, skipping...")
            break;
        
        status_text = status.full_text.replace("\n", " ")

        # Print Tweet
        print("\n", status_text, "\n")

        # Get User Input
        user_decision = input("Enter 0 for Positive, 1 for Negative, 3 for skip, or 9 to Stop: ")

        # Check User Input
        if user_decision == "9":
            print("Stopping.")
            continue_loop = False
            break;
        elif user_decision == "3":
            print("Skipping Tweet.")
        elif user_decision == "0" or user_decision == "1":
            print("Tweet assigned as: " + nn_data_categories[int(user_decision)])
            new_tweets.append([
                status_text,
                int(user_decision)
            ])
        else:
            print("Invalid Input")
    except:
        print("Tweet Error while retrieving, try again...")
        continue

# Finalize Data Collection
if len(new_tweets) != 0:
    # Ensure tweet data is up to date
    with open(data_path, "r") as f:
        tweets = json.load(f)

    for tweet in new_tweets:
        tweets.append(tweet)
    
    # Print Collection Information
    print("Total Tweets Collected: " + str(len(new_tweets)))
    print("Postive Tweets Total: " + str(len([tweet for tweet in tweets if tweet[1] == 0])))
    print("Negative Tweets Total: " + str(len([tweet for tweet in tweets if tweet[1] == 1])))

    print("Saving Tweets...")

    # Save tweets to JSON file
    with open(data_path, "w") as f:
        json.dump(tweets, f)
else:
    print("No New Tweets Collected.")
