from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import os
from sknn.mlp import Convolution, Classifier, Regressor, Layer
import numpy as np
import csv
import pickle
import random

embed_dim = 50
max_sent_len = 500
glove_path = '/mnt/share/glove.6B/glove.6B.50d.txt'
senti_path = '/mnt/share/Movie Reviews'
model_path = './movrev_senti_conv'

lm = WordNetLemmatizer()

def split(line):
    p = line.split(' ')
    return (p[0].lower(),np.array([float(f) for f in p[1:]]))

def load_glove(path):
    lines = open(glove_path).readlines()
    return dict(map(split,lines))

def encode_word(ds,w):
    w = lm.lemmatize(w.lower())

def encode_sent(glove,sent):
    words = word_tokenize(sent)
    vec = [glove[w.lower()] for w in words if w.lower() in glove]
    if len(vec) < max_sent_len:
        vec.extend([np.zeros((1,embed_dim))[0] for i in range(len(vec),max_sent_len,1)])
    elif len(vec) > max_sent_len:
        vec = vec[:max_sent_len]
    vec = np.array(vec).reshape((len(vec),embed_dim))

    #_____BAD!!!!_____#
    #vec = np.random.rand((len(vec),embed_dim)) #<-----BAD!!!!!!
    #vec = (vec+1)*100
    
    return vec

def train(nn, X_train, Y_train, path, n_epoch, save_part):
    batch = int(n_epoch/save_part)
    pickle.dump(nn,open(path,'w'))
    for i in range(batch):
        nn = pickle.load(open(path,'r'))
        nn.n_iter = save_part
        nn.fit(X_train, Y_train)
        pickle.dump(nn,open(path,'w'))
    nn.n_iter = n_epoch%save_part
    nn.fit(X_train, Y_train)
    pickle.dump(nn,open(path,'w'))

#Load sentiment analysis dataset for movie reviews
def load_ds(glove,path):
    count = 2500
    glist = os.listdir(path+'/train/pos')
    blist = os.listdir(path+'/train/neg')
    pos = np.array([encode_sent(glove,clean(' '.join(open(path+'/train/pos/'+p).readlines()))) for p in glist[:count]])
    neg = np.array([encode_sent(glove,clean(' '.join(open(path+'/train/neg/'+n).readlines()))) for n in blist[:count]])
    posY = np.zeros((pos.shape[0],2))
    posY[:,0] = 1
    negY = np.zeros((neg.shape[0],2))
    negY[:,1] = 1
    X = np.vstack((pos,neg))
    Y = np.vstack((posY,negY))
    print X.shape,'-----',Y.shape
    return (X,Y)

def test_ds(glove,nn,path):
    ptest = [path+'/test/pos/'+p for p in os.listdir(path+'/test/pos')]
    ntest = [path+'/test/neg/'+n for n in os.listdir(path+'/test/neg')]
    pX = np.array([encode_sent(glove,clean(' '.join(open(p).readlines()))) for p in ptest])
    nX = np.array([encode_sent(glove,clean(' '.join(open(n).readlines()))) for n in ntest])
    X = np.vstack((pX,nX))
    Y = nn.predict(X)
    print Y[:3]
    print Y[-3:]

def clean(x):
    return ''.join([t for t in x.strip() if ord(t) < 128])

#_________MAIN___________#
g = load_glove(glove_path)
print 'GloVe loaded...'
X,Y = load_ds(g,senti_path)

nn = Classifier(
     layers = [
        Convolution('Rectifier',channels=1,kernel_shape=(1,embed_dim)),
        Layer('Rectifier',units=400),
        Layer('Tanh',units=600),
        Layer('Softmax')],
    learning_rate = 0.001,
    verbose=True)

train(nn,X,Y,model_path,100,25)

#___TEST___#
nn = pickle.load(open(model_path,'r'))
print 'Model loaded...'
test_ds(g,nn,senti_path)
