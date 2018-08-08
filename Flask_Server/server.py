import numpy as np
import tensorflow as tf
import re
import rake
import pickle as pkl
from rake import Rake


def get_model_api():
	stopword = []
	with open("/home/anubhav/Desktop/Maui Final/SmartStoplist.txt",'r') as fil:
		alpha = fil.read().splitlines()
		for word in alpha:
			stopword.append(word.lower())
	stopword=set(stopword)
	def phrase_location(phrase,doc):
		i = doc.find(phrase)
		temp_string=doc[0:i]
		return len(temp_string.split())+1
	with open('/home/anubhav/Desktop/Maui Final/idf_score.txt','rb') as fil:
		idf_dict=pkl.load(fil)
	def rakes(raw_string):
		rake_object = rake.Rake("/home/anubhav/Desktop/Maui Final/SmartStoplist.txt", 3, 2, 1)
		rakescore={}
		keywords = rake_object.run(raw_string)
		for j in range(len(keywords)):
			rakescore[keywords[j][0]]=keywords[j][1]
		return rakescore
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
	def location(string,keywords):
		length = len(string.split())
		loc = {}
		for a in keywords:
			# print(a)
			loc[a]=phrase_location(a,string)/length
		return loc
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
		#d=semantic_scores[str(i)+'.txt']
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
		data=np.array(data)
		return data,leng,phrase_dict


	n_nodes_hl1 = 3
	n_classes = 1
	n_feature=4
	a = tf.placeholder('float',[None,None,n_feature])
	dd=tf.placeholder('int32',())

	hidden_1_layer = {'w':tf.Variable(tf.random_normal([n_feature,n_nodes_hl1])),'b':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	output = {'w':tf.Variable(tf.random_normal([n_nodes_hl1])),'b':tf.tile(tf.Variable(tf.random_normal([n_classes])),[dd])}
	l1 = tf.nn.bias_add(tf.einsum('ijk,kl->ijl',a,hidden_1_layer['w']),hidden_1_layer['b'])
	l1 = tf.nn.relu(l1)
	prediction = tf.nn.bias_add(tf.einsum('ijk,k->ij',l1,output['w']),output['b'])
	saver=tf.train.Saver()
	sess=tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess,'/home/anubhav/Desktop/Maui Final/model/file.model')
	


	def model_api(raw_string):
		lis=[]
		raw_string = re.sub('[^a-z\ ]+',' ',raw_string.lower())
		raw_string = re.sub('\ +',' ',raw_string)
		for word in raw_string.split():
			if not stopword.__contains__(word):
				lis.append(word)
		string=' '.join(lis)
		data1,max_len,phrase_dict=data(string,raw_string)
		result = sess.run(prediction,feed_dict={a:data1,dd:max_len})
		result=np.array(result)
		result = np.argsort(-1*result[0])
		final=result[:8]
		output_data=[phrase_dict[index] for index in final]
		return output_data

	return model_api