from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data import X_train, X_test, y_train, y_test

# create the pipeline for the RandomForestClassifier
model_rfc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# training the model on the training set
model_rfc.fit(X_train, y_train)

# evaluation of the model on the test set
y_pred_rf = model_rfc.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Accuracy with RandomForestClassifier: {accuracy_rf}')
