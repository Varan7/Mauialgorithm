
This is a custom model that uses some features mentioned in the Maui algorithm for keyword extraction and also uses an implementation of Rake.

The link to the paper referred: https://dl.acm.org/citation.cfm?id=1699678
Link to the Source of implementation of Rake: https://github.com/aneesha/RAKE

We used a total of 4 features including rake_score, location_score, tfidf_score, spread_score and used a neural network to calculate the weights of each of these features.

-------------------------------------------------------------------------------

Description of Folders and Files in them:

1. Flask_Server - Contains files related to the flask server created for real time hashtag recommendation

2. MauiAlgo - Contains 5 python files which implement our algorithm and a text file containing stopwords
	a. rake.py  - model for rake implementation (source mentioned above)
	b. rake1.py - imports rake and calculates the rake-score for all terms in a document
	c. maui.py  - calculates the location-score, tfidf-score, spread-score and stores them in pickle files
	d. data.py  - reads the dictionaries in the pickle files created by maui.py and trains a neural network to calculate the weight of each feature and saves the model in the folder `model`
	e. infer.py - loads the tensorflow model created by data.py and predicts the top keywords in an input document

3. model - stores the tensorflow model which contains the weights of the features
