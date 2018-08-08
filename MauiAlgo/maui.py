
#### Calculate location_score, idf_score, spread_score and tf-idf_score 

import pickle as pkl
import os
import numpy as np
import re
import pandas
import json
import sklearn
from sklearn.naive_bayes import GaussianNB
import matplotlib

keywords = []
abstracts = []
lis_names=[]
title = []
documents = []

## Read the list of documents and store them in documents
with open('/home/varan777/virtual/kp20k_training.json',mode = "r")as f :
	text = f.read()
	for i,content1 in enumerate(text.split('\n')[:-1]):
		documents.append(json.loads(content1.lower()))

content = [] #Store the list of content in the documents
stopword = []

#Load the dictionary of rakescores
with open('/home/varan777/Desktop/RakeScore.txt','rb') as fil:
	rakescore=pkl.load(fil)

for i,z in enumerate(documents):
	abstracts.append(z['abstract'])
	title.append(z['title'])
	keywords.append(z['keyword'])

# documents preprocessing
for y in range(len(abstracts)):
	alpha = abstracts[y]+' '+title[y]
	alpha = re.sub('[^a-z\ ]+',' ',alpha.lower())
	alpha = re.sub('\ +',' ',alpha)
	content.append(alpha)

# Create a list of stopwords
with open("/home/varan777/virtual/SmartStoplist.txt",'r') as fil:
	alpha = fil.read().splitlines()
	for word in alpha:
		stopword.append(word.lower())
stopword=set(stopword)


# Calculate tf-idf score
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True,ngram_range=(1,2),stop_words=stopword)
bag_of_words  = vectorizer.fit_transform(content)



idf = vectorizer.idf_

features = vectorizer.get_feature_names()

# Calculate idf-score
idf_dict={}
for i,phrase in enumerate(features):
	idf_dict[phrase]=idf[i]

#print(idf_dict)

dic10={}
for i in range(0,len(features)):
	dic10[features[i]]=i
location_score={}
tfidf_score = {}
x = content


def remove_stopword(doc,stopword):
	x2=[]
	for word in doc.split():
		if word not in stopword:
			x2.append(word)
	return(' '.join(x2))



def phrase_location(phrase,doc):
	i = doc.find(phrase)
	temp_string=doc[0:i]
	return len(temp_string.split())+1


for j in range(0,len(x)):
	print(str(j)+' t')

	x[j]=remove_stopword(x[j],stopword)
	length = len(x[j].split())
	
	tfidf_score[str(j)+'.txt']={}
	location_score[str(j)+'.txt']={}
	words = rakescore[str(j)+'.txt'].keys()
	for a in words:
		# print(a)
		location_score[str(j)+'.txt'][a]=phrase_location(a,x[j])/length
		c = dic10[a]
		b=bag_of_words[j,c]
		tfidf_score[str(j)+'.txt'][a]=b/length

			

	

spread_score = {}


for j in range(0,len(x)):
	print(str(j)+' s')
	length = len(x[j].split())
	first_occurence = {}
	last_occurence= {}
	spread_score[str(j)+'.txt'] = {}
	words = rakescore[str(j)+'.txt'].keys()
	for a in words:
		first_occurence[a] = phrase_location(a,x[j])
		bc = a
		a = a.split(" ")
		a = a[-1::-1]
		a = ' '.join(a)
		cd = x[j].split(" ")
		cd = cd[-1::-1]
		cd = ' '.join(cd)
		last_occurence[bc] =  length +1- phrase_location(a,cd)
		if(len(a.split())==1):
			spread_score[str(j)+'.txt'][bc] = (last_occurence[bc]-first_occurence[bc])/length
		else:
			ass=(last_occurence[bc]-first_occurence[bc]-1)/length 
			spread_score[str(j)+'.txt'][bc] = ass

	

with open('tfidf_score.txt','wb') as fil:
	pkl.dump(tfidf_score,fil)
with open('spread_score.txt','wb') as fil:
	pkl.dump(spread_score,fil)
with open('location_score.txt','wb') as fil:
	pkl.dump(location_score,fil)

with open('idf_score.txt','wb') as fil:
	pkl.dump(idf_dict,fil)