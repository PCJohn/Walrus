from nltk import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import os
from sknn.mlp import Convolution, Classifier, Regressor, Layer
import numpy as np
import csv
import pickle

embed_dim = 50
max_sent_len = 100
glove_path = '/mnt/share/glove.6B/glove.6B.50d.txt'
lm = WordNetLemmatizer()

def split(line):
    p = line.split(' ')
    return (p[0],np.array([float(f) for f in p[1:]]))

def load_ds(path):
    lines = open(glove_path).readlines()
    return dict(map(split,lines))

def encode_word(ds,w):
    w = lm.lemmatize(w.lower())


def encode_sent(ds,sent):
    words = word_tokenize(sent)
    vec = [ds[w.lower()] for w in words if w in ds]
    if len(vec) < max_sent_len:
        pad = np.array([0 for j in range(embed_dim)])
        vec.extend([pad.copy() for i in range(len(vec),max_sent_len,1)])
    elif len(vec) > max_sent_len:
        vec = vec[:max_sent_len]
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

def train_csv(g,nn,path,train_len):
    X = []
    Y = []
    r = csv.reader(open(path,'r'))
    r.next() #he first line is the column headings
    i = 0
    for row in r:
        i = i+1
        if i == train_len:
            break
        y = [0,0]
        y[int(row[1])] = 1
        Y.append(y)
        enc = encode_sent(g,clean(row[3]))
        X.append(enc)
    X = np.array(X)
    Y = np.array(Y)
    print X.shape
    print Y.shape
    print '______'
    train(nn,X,Y,'./senti_model',20,5)

#_________MAIN___________#
g = load_ds(glove_path)
print 'GloVe loaded...'

nn = Classifier(
     layers=[
        #Convolution('Rectifier',channels=1,kernel_shape=(3,embed_dim)),
        Layer('Rectifier',units=96),
        Layer('Rectifier',units=128),
        Layer('Rectifier',units=256),
        Layer('Rectifier',units=128),
        Layer('Rectifier',units=96),
        Layer('Softmax')],
     learning_rate=0.001,
     verbose=True)

train_csv(g,nn,'/mnt/share/Senti_csv/Sentiment Analysis Dataset.csv',100000)

#_______TESTING_________#
nn = pickle.load(open('./senti_model','r'))
while True:
    sent = raw_input('Enter text:')
    if len(sent) > 0:
        if sent == 'quit':
            break
        print nn.predict(np.array([encode_sent(g,sent)]))
