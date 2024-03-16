from sklearn.datasets import fetch_20newsgroups
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf= False)
trainvec = vectorizer.fit_transform(train.data)
testvec = vectorizer.transform(test.data)

from sklearn.naive_bayes import MultinomialNB 
classifier = MultinomialNB() 
classifier.fit(trainvec, train.target)
classes = classifier.predict(testvec)


