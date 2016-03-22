import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2

# custom tokenize func
import re
REGEX = re.compile(r",\s*")
def tokenize(text):
    tmp_lst = text.strip().lower().split(' ')
    ret_lst = []
    for w in tmp_lst:
        if w.isalpha():
            ret_lst.append(w)
    return ret_lst

#print tokenize("aa bb ccc 123")

df = pd.read_csv('datasets/SMSSpamCollection.txt', delimiter='\t', header=None)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])
vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=True, smooth_idf=False)
X_train = vectorizer.fit_transform(X_train_raw)
lst = []
idf = vectorizer.idf_
fea = vectorizer.get_feature_names()
for i in xrange(0, len(idf)):
    lst.append((fea[i], idf[i]))
lst = sorted(lst, key=lambda x:float(x[1]), reverse=True)
print lst[:50]



# ch2 = SelectKBest(chi2, k=opts.select_chi2)
# X_train = ch2.fit_transform(X_train, y_train)
# print X_train

# X_test = vectorizer.transform(X_test_raw)
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
# predictions = classifier.predict(X_test)
#
# for i, prediction in enumerate(predictions[:5]):
#     print 'Prediction %s. Message %s' % (prediction, X_test_raw[i])
