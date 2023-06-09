# Python Neural Network Tweet Classifier
Python Neural Network which Classifes Tweets between Negative and Positive sentiments.

Made after following the book, Neural Networks from Scratch by Harrison Kinsley and Daniel Kukiela

## Sentiment Tweet Data
Tweet data is from https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis.
Data is then separated into Positive and Negative tweets. Stop words are then removed.

Data inputted into the neural network is character-level one-hot encoded data. Each string is of length 280, with 44 possible characters. This requires 280*44 inputs into the neural network.

## Website Integration
With HTML 5 and Javascript fetch requests, the front-end webpage can interact with the neural network through RESTful API routes. Currently only two routes are working since the shutdown of Twitter's free v1.1 API. Because of this, only text submissions work since tweets cannot be retrieved by ID.

The website is hosted in 'main.py'.

### Homepage:
<img src="https://github.com/j-cob44/Python_NN_Tweet_Classifier/blob/main/readme_imgs/01_main.png" alt="Main Page"/>

### Evaluation Example:
<img src="https://github.com/j-cob44/Python_NN_Tweet_Classifier/blob/main/readme_imgs/02_evaluation.png" alt="Main Page"/>

After evaluating text or a tweet on the network, the user can specify if they believe the prediction to be correct. When submitted, the data is collected and stored separate to the initial data training set. This can then be used to further train the model in the future.

## Neural Network Model Creation
### neural_network.py
'Neural_network.py' contains functions for training, retraining, and hyperparameter tuning a model. 

### Training
Creates and trains a model from training data and optionally validation data. Then the model and information about it is saved into "models/"

### Re-training
Loads an old model and trains the model with a dataset. The retrained model is then saved over the old model.

## Hyperparameter Tuning
### Grid Search
Generates all possible sets of parameters and iterates through all of them. This will iterate through all sets. The process is multi-threaded and will test as many models as threads allowed. This takes significant computational resources and time to run.

### Random Search
Generates all possible sets of parameters and iterates randomly through them. This will iterate through as many sets as told to. The process is multithreaded and will test a batch of sets based on the number of threads assigned. This can take less time than a Grid Search since not every set of parameters is tested.

### Bayesian Search
By setting the lower and upper bounds of each parameter, the Bayesian search will iterate through different settings trying to improve its score. The Score is based on Accuracy - Loss in order to maximize accuracy and minimize loss. This is not multi-threaded.

Uses BayesionOptimization function from package BayesianOptimization by Fernando Nogueira at https://github.com/fmfn/BayesianOptimization


