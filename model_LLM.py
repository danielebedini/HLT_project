# Large language model for sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from data import X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced

# create a pipeline
model_llm = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# train the model
model_llm.fit(X_train_balanced, y_train_balanced)

# make predictions
y_pred = model_llm.predict(X_test_balanced)

# calculate the accuracy
accuracy = accuracy_score(y_test_balanced, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Initial accuracy with unbalanced dataset: 0.80
# Initial accuracy with balanced dataset: 0.48

print(classification_report(y_test_balanced, y_pred))