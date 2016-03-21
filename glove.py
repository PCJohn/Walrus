from nltk import word_tokenize, sent_tokenize
import os
from sknn.mlp import Convolution, Classifier, Regressor, Layer
import numpy as np
import csv
import pickle

embed_dim = 50
glove_path = '/mnt/share/glove.6B/glove.6B.50d.txt'

def split(line):
    p = line.split(' ')
    return (p[0],np.array([float(f) for f in p[1:]]))

def load_ds(path):
    lines = open(glove_path).readlines()
    return dict(map(split,lines))

def encode_sent(ds,sent):
    words = word_tokenize(sent)
    vec = [ds[w.lower()] for w in words if w in ds]
    vec = vec[:1]
    return np.array(vec).reshape((len(vec),embed_dim))

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

def clean(x):
    return ''.join([t for t in x.strip() if ord(t) < 128])

def train_csv(g,nn,path):
    X = []
    Y = []
    r = csv.reader(open(path,'r'))
    i = 0
    for row in r:
        if i == 0:
            i = 1
            continue
        print row
        y = [0,0]
        y[int(row[1])] = 1
        Y.append(y)
        enc = encode_sent(g,clean(row[3]))
        X.append(enc)

        i = i+1
        if i == 3:
            break
    for x in X:
        print x.shape,'==='
    X = np.array(X)
    Y = np.array(Y)
    print X.shape
    print Y.shape
    print '______'
    train(nn,X,Y,'./senti_model',10,5)

#_________MAIN___________#
g = load_ds(glove_path)
print 'GloVe loaded...'

nn = Classifier(
     layers=[
        Convolution('Rectifier',channels=1,kernel_shape=(1,embed_dim)),
        Layer('Rectifier',units=128),
        Layer('Softmax')],
     learning_rate=0.001,
     verbose=True)

train_csv(g,nn,'/mnt/share/Senti_csv/Sentiment Analysis Dataset.csv')

#_______TESTING_________#
#while True:
#    sent = raw_input('Enter text:')
#    if len(sent) > 0:
#        if sent == 'quit':
#            break
#        print nn.predict(np.array([encode_sent(g,sent)]))
