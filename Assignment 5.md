# Assignment 5 - Horoschope classification with scikit-learn
Firstly, the pandas module and some functions from scikit-learn are imported
```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
```
Then, the data is read using pandas
```python
data = pd.read_csv('horoscopes.csv')
```


## Task 1 - Binary classification
### Creating indeces
For task 1, we need to create indeces for the two signs used for the classification task:
```python
idxs0 = data['sign'] == 'aries'
idxs1 = data['sign'] == 'gemini'
idxs = idxs0.values + idxs1.values
```
### Structuring data
Then, x and y are defined based on the indeces just created, to ensure that only our chosen signs are used for the task:
```python
corpus = data['horoscope'].loc[idxs]
cv = CountVectorizer()
y = data['sign'].loc[idxs].values
X = cv.fit_transform(corpus.values).toarray()
```
### Splitting the data
Now, the data is split into 'train' and 'test' so that the Naive Bayes can be trained, but also tested in order to measure the accuracy. 'test-size' is set to 0.20 to create an 80/20 split between training and test-data. 'random_state' is set to 0 to make the script split the data in the same place on each run.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.20, 
    random_state=0
    )
```
### Instantiate and train
Then, a Naive Bayes is instantiated and trained based on the training datasets for both 'X' and 'y'
```python
classifier = MultinomialNB(fit_prior=True)
classifier.fit(X_train, y_train)
```
### Testing
Now, a prediction for 'y' and a confusion matrix is created:
```python
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
```
Lastly, the relative accuracy and accuracy in instances are printed:
```python
print(f"[INFO] Relative accuracy: {accuracy_score(y_test, y_pred)}")
print(f"[INFO] Accuracy in instances: {accuracy_score(y_test, y_pred, normalize=False)}")
```
### Results
When you run the script, this is printed:
```
[INFO] Relative accuracy: 0.5138888888888888
[INFO] Accuracy in instances: 222
```
Given that the zero rate is 50% the classifier is in this case not very good, as the accuracy is only slighter higher than the zero rate. This could mean that horoscopes cannot be accurately classified based on their content, or at least that a Naive Bayes might not be the right tool for such a task.

## Task 2 - Effects of preprocessing
Almost all of this task is exactly the same as in task 1. However, in this case, when structuring the data, these parameters are added to the 'CountVectorizer()' function:
- max_df=0.8
- min_df=30
- stop_words='english'

This ensures that words used very frequently or very rarely are cut out. The same goes for stopdwords
```python
cv = CountVectorizer(max_df=0.8, min_df=30, stop_words='english')
```
Other than this slight change, nothing has been changed from the previous task.
### Results
When running this script, the following is printed
```
[INFO] Relative accuracy: 0.5555555555555556
[INFO] Accuracy in instances: 240
```
As can be seen, the accuracy here is slightly higher than in the previous task. This tells us that when stopwords and words that are either very frequent or very rare are removed, it becomes easier to accurately classify horoscopes, although not much.

## Task 3 - Multilabel classification
This task largely similar to the previous one. Here, however, the indeces are created using all the signs instead of just using two of them
```python
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
```
By doing this, we train the Naive Bayes in classifying all of the signs, rather than only two.
### Results
When this script is run, this is printed:
```
[INFO] Relative accuracy: 0.11274131274131274
[INFO] Accuracy in instances: 292
```
Here, the zero rate is 8.4%, which means that it is also very difficult to classify horoscopes based on their content using Naive Bayes when you do it to all the signs. However, there is still some connection, as the accuracy did turn out slightly higher than the zero rate. 
