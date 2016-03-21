import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score

df = pd.read_csv('datasets/wine.data', sep=',')
print df.keys()

# plt.scatter(df['1'], df['2.29'])
# plt.title('Wine image')
# plt.show()
print list(df.columns)[1:]
X = df[list(df.columns)[1:]]
y = df['1']
regressor = LinearRegression()
scores = cross_val_score(regressor, X, y, cv=5)
print scores.mean(), scores
