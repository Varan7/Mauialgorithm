
# Build Tensorflow Model to predict weights for different features

import pickle as pkl
import numpy as np
import json
import tensorflow as tf

def zerolist():
	lis1 = []
	lis1.append(0)
	lis1.append(0)
	lis1.append(0)
	lis1.append(0)
	return lis1
with open("/home/varan777/virtual/tfidf_score.txt",'rb') as fil:
	tfidf_scores= pkl.load(fil)
with open("/home/varan777/virtual/spread_score.txt",'rb') as fil:
	spread_scores = pkl.load(fil)
with open("/home/varan777/virtual/location_score.txt",'rb') as fil:
	location_socres = pkl.load(fil)
with open("/home/varan777/Desktop/RakeScore.txt",'rb') as fil:
	rake_scores = pkl.load(fil)

data1 =[]
with open('/home/varan777/virtual/kp20k_training.json',mode = "r")as f :
	text = f.read()
	for content1 in text.split('\n')[:-1]:
		data1.append(json.loads(content1.lower()))
keys = []
keywords=[]
for z in data1:
	keywords.append(z['keyword'])
for y in range(len(keywords)):
	keys.append(''.join(keywords[y]))

max_len=0
for i in range(len(tfidf_scores.keys())):
	e=rake_scores[str(i)+'.txt']
	max_len=max(max_len,len(e.keys()))

def datas(p):
	data = []
	data_keys=[]
	global tfidf_scores
	global spread_scores
	global location_socres
	global rake_scores
	if p==5:
		l=500000
		m=len(tfidf_scores.keys())
	else:
		l=p*100000
		m=(p+1)*100000
	for i in range(l,m):
		a=tfidf_scores[str(i)+'.txt']
		b=spread_scores[str(i)+'.txt']
		c=location_socres[str(i)+'.txt']
		e=rake_scores[str(i)+'.txt']
		lis =[]
		phrase_list = set(keys[i].split(';'))
		ar = np.zeros(max_len)
		for j,phrase in enumerate(e.keys()):
			if phrase in phrase_list:
				ar[j]=1
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
		for _ in range(len(e.keys()),max_len):
			lis.append(zerolist())
		data.append(lis)
		data_keys.append(ar)
	data= np.array(data)
	data_keys=np.array(data_keys)
	return data,data_keys


n_nodes_hl1 = 3
n_classes = 1
batch_size = 64
nm_epcohs =10
n_feature=4

a = tf.placeholder('float',[None,max_len,n_feature])
b = tf.placeholder('float',[None,max_len])

def neural_network_model(data,max_len):
	hidden_1_layer = {'w':tf.Variable(tf.random_normal([n_feature,n_nodes_hl1])),'b':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	output = {'w':tf.Variable(tf.random_normal([n_nodes_hl1])),'b':tf.tile(tf.Variable(tf.random_normal([n_classes])),[max_len])}
	
	l1 = tf.nn.bias_add(tf.einsum('ijk,kl->ijl',data,hidden_1_layer['w']),hidden_1_layer['b'])
	l1 = tf.nn.relu(l1)

	out = tf.nn.bias_add(tf.einsum('ijk,k->ij',l1,output['w']),output['b'])
	return out

def train_nn(max_len):
	dataset = tf.data.Dataset.from_tensor_slices((a,b))
	dataset=dataset.repeat(nm_epcohs)
	dataset=dataset.shuffle(666)
	batched_dataset=dataset.batch(batch_size)
	iterator=batched_dataset.make_initializable_iterator()
	next_element=iterator.get_next()
	prediction = neural_network_model(next_element[0],max_len)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction,labels=next_element[1]))
	optimizer = tf.train.AdamOptimizer(0.0005).minimize(cost)
	saver=tf.train.Saver()

	with tf.Session() as sess:
		print('yay')
		sess.run(tf.global_variables_initializer())
		for epoch in range(1,nm_epcohs+1):
			epoch_loss = 0
			for i in range(6):
				k,l = datas(i)
				sess.run(iterator.initializer,feed_dict={a:k,b:l})
				n = np.shape(k)[0] 
				for _ in range(int(n/batch_size)):
					_,c = sess.run([optimizer,cost])
					epoch_loss += c
				print("Epoch",epoch,'completed out of',nm_epcohs,'loss:',epoch_loss)
		saver.save(sess,'/home/varan777/model/file.model')
		
train_nn(max_len)				
	


