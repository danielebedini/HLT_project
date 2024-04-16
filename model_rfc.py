from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class RandomForestModelBuilder:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier())
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy with RandomForestClassifier: {accuracy}')
        print(classification_report(y_test, y_pred,zero_division=0))

    def get_model(self):
        return self.model
