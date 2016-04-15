import os
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from sknn.mlp import Convolution, Layer, Classifier, Regressor

phrase_path = './Data/SSTb/dictionary.txt'
glove_path = './Data/glove.6B/glove.6B.50d.txt'
sent_path = './Data/SSTb/sentiment_labels.txt'
model_path = './SSTb_model'

lm = WordNetLemmatizer()

WORD_DIM = 50
PHRASE_LEN = 40

NEG = 0
NET = 1
POS = 2

def clean(line):
	return ''.join([c for c in line if ord(c) < 128])

def split(line):
	p = line.split(' ')
	return (p[0].lower(),np.array([float(f) for f in p[1:]]))

def load_glove():
	return dict(map(split,open(glove_path).readlines()))

def encode_phrase(glove,phrase):
	em = np.array([])
	for w in word_tokenize(phrase)[:PHRASE_LEN]:
		w = lm.lemmatize(w.lower())
		if w in glove:
			if em.size == 0:
				em = np.reshape(glove[w],(1,WORD_DIM))
			else:
				em = np.vstack( (em,(np.reshape(glove[w],(1,WORD_DIM)))) )
	if em.size == 0:
		return None
	while em.shape[0] < PHRASE_LEN:
		em = np.vstack( (em,np.zeros(shape=(1,WORD_DIM))) )
	return np.array(em)

def load_ds():
	glove =	load_glove()
	print 'GloVe loaded...'
	
	X = []
	Y = []

	p_id = {}
	p_sent = {}
	for line in open(phrase_path).readlines():
		line = clean(line).strip()
		p = line.split('|')
		p_id[int(p[1])] = p[0]
	for line in open(sent_path).readlines()[1:]:
		line = clean(line).strip()
		p = line.split('|')
		p_sent[int(p[0])] = float(p[1])

	for id in p_sent.keys():
		em = encode_phrase(glove,p_id[id])
		if not em is None:
			X.append(em)
		else:
			del p_id[id]
			del p_sent[id]
		sent = NEG
		if id in p_sent:
			if p_sent[id] < 0.5:
				sent = NEG
			elif p_sent[id] >= 0.5:
				sent = POS
			out = np.array([0,0,0])
			out[sent] = 1
			Y.append(out)
	
	return (np.array(X),np.array(Y))

def train(X,Y):
	print X.shape
	print Y.shape
	nn = Classifier(
		layers = [
			Convolution('Rectifier',channels=1,kernel_shape=(5,WORD_DIM)),
			Layer('Rectifier',units=300),
			Layer('Rectifier',units=300),
			Layer('Softmax')],
		learning_rate = 0.001,
		verbose = True)
	print 'Net created...'
	#nn.n_iter = 150
	nn.fit(X,Y)
	pickle.dump(nn,open(model_path,'w'))
	print 'Model saved'

def test(path):
	pass

x,y = load_ds()
print x.shape,'==',y.shape
train(x,y)
test(model_path)
