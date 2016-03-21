from __future__ import division
from sknn.mlp import Regressor, Layer
import numpy as np
from matplotlib import pyplot as plt

LIM = 628

X_train = np.array([i/LIM for i in range(LIM)])
Y_train = np.array([np.sin(x*LIM/100.0) for x in X_train])
plt.plot(X_train,Y_train,'ro')
plt.show()
X_train = X_train.reshape((X_train.size,1))
Y_train = Y_train.reshape((Y_train.size,1))

nn = Regressor(
    layers=[
        Layer("Linear",units=1),
        Layer("Tanh", units=10),
        Layer("Tanh",units=10),
        Layer("Linear", units=1)],
    learning_rate=0.001,
    n_iter=1000)
nn.fit(X_train, Y_train)

X_test = np.array([i/LIM for i in range(LIM)])
X_test = X_test.reshape((X_test.size,1))
Y_test = nn.predict(X_train)

plt.plot([x[0] for x in X_test],[y[0] for y in Y_test],'ro')
plt.show()
