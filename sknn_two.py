from __future__ import division
from sknn.mlp import Regressor, Layer, Classifier, Convolution
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
import cv2

SIZE_R = 64
SIZE_C = 64
CAT = ['airplanes','Faces_easy','Motorbikes']
TRAIN_LEN = 250
TEST_LEN = 20
CAT_COUNT = len(CAT)

def load_img():
    path = '/mnt/share/101 Object Categories/'
    img_list = []
    test_list = []
    cat_list = []
    c = 0
    for cat in CAT:
        category = os.listdir(path+cat)
        for i in category[:TRAIN_LEN]:
            img = np.array(cv2.imread(path+cat+'/'+i,0))
            img = cv2.resize(img,(SIZE_R,SIZE_C))
            print cat,'--',i,'--',c
            img_list.append(img)
            cat_v = [0 for i in range(CAT_COUNT)]
            cat_v[c] = 1
            cat_list.append(cat_v)
        
        for i in category[-TEST_LEN:]:
            img = np.array(cv2.imread(path+cat+'/'+i,0))
            img = cv2.resize(img,(SIZE_R,SIZE_C))
            test_list.append(img)
        c = c+1
    img_list = np.array(img_list)
    cat_list = np.array(cat_list)
    test_list = np.array(test_list)
    return (img_list.size,img_list,cat_list,test_list)

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

N,img,cat,test = load_img()
print 'Images loaded'
print '____________'
X_train = img
Y_train = cat
X_test = test

print N
print X_train.shape,' -- ',X_test.shape
print Y_train.shape
print '____________'

print X_train[:5]
print Y_train[:5]
print Y_train[-5:]

nn = Classifier(
    layers=[
        Convolution("Rectifier",channels=48,kernel_shape=(5,5),pool_shape=(3,3)),
        Convolution("Rectifier",channels=96,kernel_shape=(11,11),pool_shape=(5,5)),
        Layer("Rectifier",units=128),
        Layer("Rectifier",units=128),
        Layer("Softmax")],
    learning_rate=0.001,
    verbose=True)

#train(nn, X_train, Y_train, './model', n_epoch=100, save_part=5)
#train(nn, X_train, Y_train, './skt_again', n_epoch=100, save_part=5)

#-----TESTING-----#
nn = pickle.load(open('./skt_again','r'))
print 'Network loaded'
print nn
print '--------'

pred = nn.predict(test)
for i,p in enumerate(pred):
    #plt.imshow(test[i])
    #plt.show()
    print p,'<='
