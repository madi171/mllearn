from sklearn.feature_extraction.text import HashingVectorizer

corpus = {'ass', 'bdfs', 'cer', 'dsssdf'}
vectorizer = HashingVectorizer(n_features=14)
print vectorizer.transform(corpus).todense()
