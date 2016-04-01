import sys
import numpy as np
import math


def log(s):
    print >>  sys.stderr, "LOG: %s" % s


# def sigmoid()
def sigmoid(x):
    return 1.0 /(1.0 + np.exp(-x))


def load_dataset(top=None, sample=None):
    fobj = open("datasets/dac_sample.txt", 'r')
    label = []
    data = []
    for line in fobj:
        ss = line.split('\t')
        data.append([sigmoid(float(x)) if x != '' else 0.0 for x in ss[1:14]])
        label.append(float(ss[0]))
    return np.mat(data[0:80000]), np.mat(label[0:80000]).transpose(), np.mat(data[80001:]), np.mat(label[80001:]).transpose()


# load datasets, y_train, X_train
MAX_FEATURE_DIM = 13
MAX_ITER = 50

log("Start to load dataset")
X_train, y_train, X_test, y_test = load_dataset()
log("Done to load dataset")

# init theta(random)
#theta = np.asmatrix(np.random.randn(MAX_FEATURE_DIM, 1))
theta = np.asmatrix(np.ones((MAX_FEATURE_DIM, 1)))
#theta = np.asmatrix(np.zeros((MAX_FEATURE_DIM, 1)))
alpha = 0.00002
option = "gd"
log("Begin to train using %s, init theta=%s" % (option, theta))

if option == "gd":
    # iter to train theta
    for iter in range(MAX_ITER):
        error = (y_train - sigmoid(X_train * theta))
        grad = X_train.transpose() * error
        theta = theta + alpha * grad
        log("\tstart %d iteration... grad=%s theta=%s" % (iter, grad.transpose(), theta.transpose()))
elif option == "sgd":
    # iter to train theta using SGD
    for iter in range(MAX_ITER):
        for i in range(len(X_train)):
            error = (y_train[i] - sigmoid(X_train[i] * theta))
            grad = X_train[i].transpose() * error
            theta = theta + alpha * grad
        log("\tstart %d iteration... grad=%s theta=%s" % (iter, grad.transpose(), theta.transpose()))

# predict
correct_sum = 0
pred = sigmoid(X_test * theta)
r = zip(y_test, [1.0 if p > 0.5 else 0.0 for p in pred])
rr = [1 if (x[0] == x[1]) else 0 for x in r]
print "correct rate=%.2f%%" % (100.0 * sum(rr) / len(rr))