from __future__ import division
from sknn.mlp import Regressor, Layer, Classifier, Convolution
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import cv2
import math

def is_prime(N):
    for i in range(1,int(N**0.5)+1,1):
        if (i > 1) & (N%i == 0):
            return False
    return True

def prime_list(N):
    return [n for n in range(2,N,1) if is_prime(n)]

def create_ds(N):
    primes = prime_list(N)
    size = int(2*math.log(N)+1)
    X_train = []
    Y_train = []
    for p1 in primes:
        for p2 in primes:
            d = np.array([int(c) for c in str(p1*p2)])
            d = np.append([0 for i in range(size-d.size)],d)
            X_train.append([d])
            if p1 < p2:
                Y_train.append([1/p1])
            else:
                Y_train.append([1/p2])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    return (X_train,Y_train)

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


N = 100
X,Y = create_ds(N)
print X.shape,'--',Y.shape
print X[:5]
print Y[:5]
print '___________'
nn = Regressor(
    layers=[
        #Convolution("Rectifier",channels=1,kernel_shape=(1,1)),
        Layer("Rectifier",units=128),
        Layer("Rectifier",units=128),
        Layer("Linear",units=64),
        Layer("Tanh")],
    learning_rate=0.01,
    verbose=True)
train(nn, X, Y, './mod_prim', 2, 2)

print "#-----TESTING-----#"
nn = pickle.load(open('./mod_prim','r'))
test = create_ds(2*N)
pred = nn.predict(test)
for i,p in enumerate(pred):
    #plt.imshow(test[i])
    #plt.show()
    print test[i],' == ',round(1/p)
