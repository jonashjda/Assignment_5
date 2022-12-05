import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# read data
data = pd.read_csv('horoscopes.csv')

# create indices
idxs0 = data['sign'] == 'aries'
idxs1 = data['sign'] == 'taurus'
idxs2 = data['sign'] == 'gemini'
idxs3 = data['sign'] == 'cancer'
idxs4 = data['sign'] == 'leo'
idxs5 = data['sign'] == 'virgo'
idxs6 = data['sign'] == 'libra'
idxs7 = data['sign'] == 'scorpio'
idxs8 = data['sign'] == 'sagittarius'
idxs9 = data['sign'] == 'capricorn'
idxs10 = data['sign'] == 'aquarius'
idxs11 = data['sign'] == 'pisces'
idxs = idxs0.values + idxs1.values + idxs2.values + idxs3.values + idxs4.values + idxs5.values + idxs6.values + idxs7.values + idxs8.values + idxs9.values + idxs10.values + idxs11.values


# structuring text data
corpus = data['horoscope'].loc[idxs]
cv = CountVectorizer(max_df=0.7, min_df=30, stop_words='english')
y = data['sign'].loc[idxs].values
X = cv.fit_transform(corpus.values).toarray()

# training-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# instantiate and train Naive Bayes
classifier = MultinomialNB(fit_prior=True)
classifier.fit(X_train, y_train)

# test model
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(f"[INFO] Relative accuracy: {accuracy_score(y_test, y_pred)}")
print(f"[INFO] Accuracy in instances: {accuracy_score(y_test, y_pred, normalize=False)}")
