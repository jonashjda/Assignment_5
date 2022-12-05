import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import random

# read data
data = pd.read_csv('horoscopes.csv')

# create indices
idxs0 = data['sign'] == 'aries'
idxs1 = data['sign'] == 'gemini'
idxs = idxs0.values + idxs1.values

# structuring text data
corpus = data['horoscope'].loc[idxs]
cv = CountVectorizer()
y = data['sign'].loc[idxs].values
X = cv.fit_transform(corpus.values).toarray()

# training-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.20, 
    random_state=0
    )

# instantiate and train Naive Bayes
classifier = MultinomialNB(fit_prior=True)
classifier.fit(X_train, y_train)

# test model
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(f"[INFO] Relative accuracy: {accuracy_score(y_test, y_pred)}")
print(f"[INFO] Accuracy in instances: {accuracy_score(y_test, y_pred, normalize=False)}")