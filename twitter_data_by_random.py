# Twitter_Data_By_Random.py - Grabs a random recent Tweet for Analysis
# Jacob Burton 2023

import tweepy
import json
import numpy as np

from __authsecrets__ import secret
from twitter_data import nn_data_categories

# Data path (Change when creating different datasets)
data_path = "tweet_data/training/nn_tweet_data.json"

# Dictionary for Tweet Categories to Search
tweet_categories = {
    0: "#politics",
    1: "#sports",
    2: "#news",
    3: "baby",
    4: "new",
    5: "#technology",
    6: "beautiful",
    7: "#movies",
    8: "#gaming",
    9: "#youtube",
    10: "friend",
    11: "#twitter",
    12: "#facebook",
    13: "#anime",
    14: "inflation",
    15: "gaming",
    16: "gamer",
    17: "Call of Duty",
    18: "Fortnite",
    19: "Minecraft",
    20: "League of Legends",
    21: "Valorant",
    22: "@WhiteHouse",
    23: "@POTUS",
    24: "@JoeBiden",
    25: "Trump",
    26: "Climate",
    27: "economy",
    28: "tiktok",
    29: "youtuber",
    30: "hate",
    31: "you",
    32: "love",
    33: "kys",
    34: "happy",
    35: "exciting",
    36: "good",
    37: "bad",
    38: "angry",
    39: "fight",
    40: "stan",
    41: "kill",
    42: "die",
    43: "space",
    44: "beauty",
    45: "nature",
    46: "science",
    47: "amazing",
    48: "funny",
    49: "breathtaking",
    50: "cute",
    51: "NASA",
    52: "elon musk",
    53: "cool",
    54: "sweet"
}

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
previous_query = ""
query = ""
while continue_loop:
    query = np.random.randint(0, len(tweet_categories))

    # If Query is the Same as Previous, retry
    if query != previous_query:
        # Get Tweets from Twitter API
        try:
            for tweet in tweepy.Cursor(api.search_tweets, q=tweet_categories[query], result_type="recent", lang="en", count=1).items(1):
                # Skip Retweets
                if tweet.retweeted or 'RT @' in tweet.text:
                    print("Skipping Retweet... from category " + tweet_categories[query])
                    break;

                # Get Full Tweet Text / Skip if Error
                try:
                    status = api.get_status(tweet.id, tweet_mode='extended')
                except:
                    print("Tweet Error, Retrying...")
                    break
                
                # Get Tweet Text
                status_text = status.full_text.replace("\n", " ")

                # Check if duplicate new tweet
                if status_text in [tweet[0] for tweet in new_tweets]:
                    print("Skipping Tweet that has already been added...")
                    break;

                # Print Tweet
                print("Search Category: " + tweet_categories[query])
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
            
                # Print Collection Information
                print("Total Tweets Collected: " + str(len(new_tweets)))

                previous_query = query
        except Exception as e:
            print(e)
            break;

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