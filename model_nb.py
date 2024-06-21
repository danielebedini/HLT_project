# Naive bayes model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from utils import metrics_with_three_classes
from data import DataPreprocessor
# Naive bayes model

class NaiveBayesModelBuilder:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', MultinomialNB())
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

    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced, X_test_real, y_test_real
    from utils import plot_confusion_matrix

    model_nb = NaiveBayesModelBuilder()
    model_nb.train(X_train, y_train)

    
    """print("Test (difficult version) on balanced data")
    model_nb.evaluate(X_test_balanced, y_test_balanced)
    plot_confusion_matrix(model_nb.model, X_test_balanced, y_test_balanced, 'Naive Bayes Balanced Data')

    print("Test (difficult version) on unbalanced data")
    model_nb.evaluate(X_test, y_test)
    plot_confusion_matrix(model_nb.model, X_test, y_test, 'Naive Bayes Unbalanced Data')

    print("Test (difficult version) on unbalanced data")
    model_nb.evaluate(X_test_real, y_test_real)
    plot_confusion_matrix(model_nb.model, X_test_real, y_test_real, 'Naive Bayes Real Test')"""
    

    print("Test on balanced data")
    metrics_with_three_classes(model_nb.model, X_test_balanced, y_test_balanced, 'Naive Bayes Balanced Data')

    #print("Test on unbalanced data")
    #metrics_with_three_classes(model_nb.model, X_test, y_test, 'Naive Bayes Unbalanced Data')

    print("Test on real data")
    metrics_with_three_classes(model_nb.model, X_test_real, y_test_real, 'Naive Bayes Real Test')
