# Naive bayes model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from utils import metrics_with_three_classes
from data import DataPreprocessor
# Naive bayes model

class NaiveBayesModelBuilder:
    def __init__(self, alpha=1.0):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB(alpha=alpha))
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'F1 score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.model
    

if __name__ == '__main__':
    from model_nb import NaiveBayesModelBuilder
    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced


    model_nb = NaiveBayesModelBuilder()
    model_nb.train(X_train, y_train)
    metrics_with_three_classes(model_nb.model, X_test, y_test, 'Naiive Bayes Unbalanced Data')
    metrics_with_three_classes(model_nb.model, X_test_balanced, y_test_balanced, 'Naiive Bayes Balanced Data')

