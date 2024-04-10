from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from data import data, X_train, X_test, y_train, y_test

# here, we can tune the hyperparameters of the TfidfVectorizer
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8)

# creation of a pipeline that includes the TfidfVectorizer and LinearSVC
model_lsvc = Pipeline([
    ('tfidf', tfidf),
    ('clf', LinearSVC())
])

# train the model on the training set
model_lsvc.fit(X_train, y_train)

# evaluate the model on the test set
y_pred = model_lsvc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# visualize the classification report
print(classification_report(y_test, y_pred))
