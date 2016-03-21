from __future__ import division
from sknn.mlp import Regressor, Layer, Classifier, Convolution
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import cv2
import random
import subprocess as sp
import signal

SIZE_R = 64
SIZE_C = 64

def load_song(path):
    OUT_SAMPLE = 500#44100
    command = ['ffmpeg',
	           '-i',path,#'/mnt/share/music/The Beatles/Abbey Road/Mean Mr Mustard.mp3',
	           '-f','s16le',
               '-t','8',
	           '-acodec','pcm_s16le',
	           '-ar',str(OUT_SAMPLE), #Output frequency = 44100
	           '-ac','2', #Stereo - Set to 1 for mono
	           '-']
    
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    pipe.stdout.read(OUT_SAMPLE*8)
    raw_audio = pipe.stdout.read(OUT_SAMPLE*8)
    
    pipe.wait()
    audio = np.fromstring(raw_audio,dtype='int16')
    audio = audio.reshape((len(audio)/2,2))
    
    return audio

def load_song_ds(path):
    X = []
    Y = []
    categories = os.listdir(path)[:2]
    cat_count = len(categories)
    cat_map = dict([(cat,i) for i,cat in enumerate(categories)])
    for c,genre in enumerate(categories):
        cat = [0 for i in range(cat_count)]
        cat[c] = 1
        samples = os.listdir(path+genre)[:-5]
        random.shuffle(samples)
        for song in samples:
            audio = load_song(path+genre+'/'+song)
            X.append(audio)
            Y.append(cat)
    X = np.array(X)
    Y = np.array(Y)
    return (X,Y,cat_map)

def load_test_ds(path):
    X = []
    N = []
    for song in os.listdir(path):
        audio = load_song(path+'/'+song)
        X.append(audio)
        N.append(song)
    return (N,np.array(X))

def save_song(path,save_path):
    pass

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

train_path = '/mnt/share/Music genres/'
test_path  = '/mnt/share/Music genre test'
X,Y,cat_map = load_song_ds(train_path)
N_test,X_test = load_test_ds(test_path)

print 'Dataset loaded'
print X.shape
print Y.shape
print X_test.shape
print '_________________'

nn = Classifier(
    layers=[
        Convolution("Rectifier",channels=96,kernel_shape=(3,1)),
        Convolution("Rectifier",channels=120,kernel_shape=(5,1)),
        Layer("Rectifier",units=128),
        Layer("Rectifier",units=128),
        Layer("Softmax")],
    learning_rate=0.0001,
    verbose=True)

train(nn, X, Y, './model_music', n_epoch=100, save_part=5)

#-----TESTING-----#
nn = pickle.load(open('./model_music','r'))
print 'Network loaded'
print nn
print '--------'

print cat_map
pred = nn.predict(X_test)
for i,p in enumerate(pred):
    print N_test[i],'--',p
