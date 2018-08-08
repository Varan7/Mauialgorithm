
## Recommend Hashtags for an input text file

import pickle as pkl
import numpy as np
import json
import tensorflow as tf
import re
import rake
from rake import Rake
import time


start=time.time()

# Read and preprocess the input text file
## raw_tring stores the entire text file while string stores the words excluding stopwords

with open('/home/anubhav/Desktop/text.txt','r') as fil:
	alpha = fil.read()
	alpha = re.sub('[^a-z\ ]+',' ',alpha.lower())
	alpha = re.sub('\ +',' ',alpha)
	raw_string = alpha
stopword = [] # list of stopwords

with open("/home/anubhav/Desktop/Maui Final/SmartStoplist.txt",'r') as fil:
	alpha = fil.read().splitlines()
	for word in alpha:
		stopword.append(word.lower())
stopword=set(stopword)
lis = []
for word in raw_string.split():
	if not stopword.__contains__(word):
		lis.append(word)
		string=' '.join(lis)

# Function to find phrase location of 'phrase' in the document 'doc'
def phrase_location(phrase,doc):
	i = doc.find(phrase)
	temp_string=doc[0:i]
	return len(temp_string.split())+1

# Load the idf_score dictionary
with open('/home/anubhav/Desktop/Maui Final/idf_score.txt','rb') as fil:
	idf_dict=pkl.load(fil)

# Function to find the rake score of raw_string i.e the one including the stopwords
def rakes(raw_string):
	rake_object = rake.Rake("/home/anubhav/Desktop/Maui Final/SmartStoplist.txt", 3, 2, 1)
	rakescore={}
	keywords = rake_object.run(raw_string)
	for j in range(len(keywords)):
		rakescore[keywords[j][0]]=keywords[j][1]
	#print(rakescore.keys())
	return rakescore

# Function to find the tf-idf score of string i.e the input excluding stopwords
def tf_idf(string):
	tf1 = {}
	for word in string.split():
		if tf1.__contains__(word):
			tf1[word]+=1
		else:
			tf1[word]=1
	lis = string.split()
	for i in range(len(lis)):
		if i == len(lis)-1:
			break
		word = lis[i]+' '+lis[i+1]
		if tf1.__contains__(word):
			tf1[word]+=1
		else:
			tf1[word]=1
	for word in tf1.keys():
		if not idf_dict.__contains__(word):
			idf_dict[word]=13.48129
		tf1[word]=tf1[word]*idf_dict[word]
	return tf1

# Function to find location score of keyword in string
def location(string,keywords):
	length = len(string.split())
	loc = {}
	for a in keywords:
		# print(a)
		loc[a]=phrase_location(a,string)/length
	return loc

# Function to calculate spread score of keyword in string
def spread(string,keywords):
	length = len(string.split())
	dic2 = {}
	for a in keywords:
		v1 = phrase_location(a,string)
		bc = a.split(" ")
		bc = bc[-1::-1]
		bc = ' '.join(bc)
		cd = string.split(" ")
		cd = cd[-1::-1]
		cd = ' '.join(cd)
		v2 =  length +1- phrase_location(bc,cd)
		if(len(a.split())==1):
			dic2[a] = (v2-v1)/length
		else:
			dic2[a]=(v2-v1-1)/length 
	return dic2


def data(string,raw_string):
	phrase_dict={}
	data =[]
	e=rakes(raw_string)
	keywords=e.keys()
	
	if len(keywords)<20:
		keywords2=tf_idf(string).keys()
		for word in keywords2:
			if not keywords.__contains__(word):
				e[word]=0.0
		leng=len(keywords)
		
	

	leng = len(e.keys())
	


	a=tf_idf(string)
	b=spread(string,keywords)
	c=location(string,keywords)
	lis =[]
	
	for j,phrase in enumerate(e.keys()):
		phrase_dict[j]=phrase
		a1=a[phrase]
		b1=b[phrase]
		c1=c[phrase]
		e1=e[phrase]
		lis1 = []
		lis1.append(a1)
		lis1.append(b1)
		lis1.append(c1)
		lis1.append(e1)
		lis.append(lis1)
	data.append(lis)
	return data,leng,phrase_dict

data,max_len,phrase_dict=data(string,raw_string)
data= np.array(data)

n_nodes_hl1 = 3
n_classes = 1
n_feature=4

a = tf.placeholder('float',[None,None,n_feature])
dd=tf.placeholder('int32',())
def neural_network_model(data,dd):

	hidden_1_layer = {'w':tf.Variable(tf.random_normal([n_feature,n_nodes_hl1])),'b':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	output = {'w':tf.Variable(tf.random_normal([n_nodes_hl1])),'b':tf.tile(tf.Variable(tf.random_normal([n_classes])),[dd])}
	
	l1 = tf.nn.bias_add(tf.einsum('ijk,kl->ijl',data,hidden_1_layer['w']),hidden_1_layer['b'])
	l1 = tf.nn.relu(l1)

	out = tf.nn.bias_add(tf.einsum('ijk,k->ij',l1,output['w']),output['b'])
	return out



sess=tf.Session()
prediction = neural_network_model(a,dd)
def load_model():
	saver=tf.train.Saver()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,'/home/anubhav/Desktop/Maui Final/model/file.model')


def train_nn(data,phrase_dict):
	result = sess.run(prediction,feed_dict={a:data,dd:max_len})
	result=np.array(result)
	result = np.argsort(-1*result[0])
	final=result[:8]
	for index in final:
		print(phrase_dict[index]+'\n')
	sess.close()
		
load_model()
train_nn(data,phrase_dict)				
	
print(time.time()-start)

