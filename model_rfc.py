from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data import X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced

# create the pipeline for the RandomForestClassifier
model_rfc = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

# training the model on the training set
model_rfc.fit(X_train_balanced, y_train_balanced)

# evaluation of the model on the test set
y_pred_rf = model_rfc.predict(X_test_balanced)
accuracy_rf = accuracy_score(y_test_balanced, y_pred_rf)
print(f'Accuracy with RandomForestClassifier: {accuracy_rf}')
