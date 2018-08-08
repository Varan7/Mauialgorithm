
## Calculate the rake score for each term in all documents

from __future__ import division
import json
import os
import glob
import time
import re
import math
import operator
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk import ngrams
import rake
from rake import Rake
import pickle as pkl
stemmer = PorterStemmer()

####################### TRAINING ###############################

start = time.time()


content=[]  # To store the list of documents 
allkeywords=[] # To store the list of keywords in all documents
data = []  # To store the list of documents

# Open and read the documents
with open('/home/varan777/virtual/kp20k_training.json',mode = "r")as f :
	text = f.read()
	for content1 in text.split('\n')[:-1]:  #spilt the json file document-wise
		data.append(json.loads(content1.lower())) 
		# data now stores the list of documnets
	
# Extract the abstract, title and keyword in each training document and store them in lists
abstracts = []
title = []
keywords1 = []
txtfiles=[]  # List of name of the documents
for i,x in enumerate(data):
	abstracts.append(x['abstract'])
	title.append(x['title'])
	keywords1.append(x['keyword'])

num=0

# Preprocess the text in the documents
for i in range(len(title)):
		txtfiles.append(str(num)+'.txt')
		num+=1
		alpha = abstracts[i] + ' '+title[i]
		beta = re.sub('[^a-z\ ]+',' ',alpha.lower())
		beta = re.sub('\ +',' ',beta)
		content.append(beta)
		allkeywords.append(keywords1[i].split(';'))
print(len(title),len(content),len(keywords1),len(allkeywords),num)





print(time.time()-start)

print ("Data retrieved and pre-processed")




# Create a rake object which will calculate rake skore for each term in the document passed as parameter to it
# It will only calculate rake scores for words with length at least 3, and with a frequency of at least 1 in the document
# It will also use 2-grams for candidate keyphrases and hence calculate rake score of all possible 2-grams and 1-grams
rake_object = rake.Rake("/home/varan777/virtual/SmartStoplist.txt", 3, 2, 1)

# Dictionary to store rake score of each candidate keyphrase in all documents
rakescore={}

for i in range(len(content)): 

	keywords = rake_object.run(content[i]) #This will return a dictionary with cadidate keywords as keys and their rake score as values
	print(txtfiles[i])

	rakescore[txtfiles[i]]={} # Dictionary within the dictionary named rakescore to store the keyword and rake score in ith document
	
	for j in range(len(keywords)):
		rakescore[txtfiles[i]][keywords[j][0]]=keywords[j][1] #Store the rake score of jth keyword in ith document

	
# Check
print(rakescore[txtfiles[0]])


# Save the dictionary as pickle file
with open("/home/varan777/Desktop/RakeScore.txt",'wb') as fil:
	pkl.dump(rakescore,fil)
