import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

data = load_boston()
# print data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
X_scaler = StandardScaler()
y_scaler = StandardScaler()
# print X_train
X_train = X_scaler.fit_transform(X_train)
y_train = y_scaler.fit_transform(y_train)
# print X_train
X_test = X_scaler.fit_transform(X_test)
y_test = y_scaler.fit_transform(y_test)

regressor = SGDRegressor(loss='squared_loss')
scores = cross_val_score(regressor, X_train, y_train, cv=5)
print X_train.shape
print "CV ", scores
print regressor.fit_transform(X_train, y_train).shape
print "Test r-ss", regressor.score(X_test, y_test)
