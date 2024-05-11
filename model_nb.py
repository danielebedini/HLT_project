# Naive bayes model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from utils import plot_confusion_matrix
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
    from data import DataPreprocessor

    # load, preprocess and split balanced data
    preprocessor = DataPreprocessor('new_balanced_data.csv')
    preprocessor.load_and_preprocess()
    preprocessor.split_data()
    preprocessor.oversample()
    X_train_balanced, X_val_balanced, X_test_balanced, y_train_balanced, y_val_balanced, y_test_balanced = preprocessor.get_train_val_test_data()

    unbalanced_data = DataPreprocessor('amazon_reviews.csv')
    unbalanced_data.load_and_preprocess()
    unbalanced_data.split_data()
    unbalanced_data.oversample()
    X_train_unbalanced, X_val_unbalanced, X_test_unbalanced, y_train_unbalanced, y_val_unbalanced, y_test_unbalanced = unbalanced_data.get_train_val_test_data()

    model_nb = NaiveBayesModelBuilder()
    model_nb.train(X_train_balanced, y_train_balanced)
    model_nb.evaluate(X_val_unbalanced, y_val_unbalanced)
    #model_nb.evaluate(X_test_unbalanced, y_test_unbalanced)

    plot_confusion_matrix(model_nb.get_model(), X_val_unbalanced, y_val_unbalanced, 'Naive Bayes')
    #plot_confusion_matrix(model_nb.get_model(), X_test_unbalanced, y_test_unbalanced, 'Naive Bayes')