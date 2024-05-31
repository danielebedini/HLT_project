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
    #from data import DataPreprocessor
    from data_2 import X_train, X_test, y_train, y_test

    model_nb = NaiveBayesModelBuilder()

    '''
    data_preprocessor = DataPreprocessor('balanced_train_data.csv')
    data_preprocessor.load_and_preprocess()
    data_preprocessor.split_data(test_size=0.1, validation_size=0.1, random_state=42, stratify_column='overall')
    data_preprocessor.oversample()
    X_train_balanced, _, _, y_train_balanced, _, _ = data_preprocessor.get_train_val_test_data()

    data_preprocessor = DataPreprocessor('unbalanced_test_data.csv')
    data_preprocessor.load_and_preprocess()
    data_preprocessor.split_data(test_size=0.6, validation_size=0.2, random_state=42, stratify_column='overall')
    _, X_val_unbalanced, X_test_unbalanced, _, y_val_unbalanced, y_test_unbalanced = data_preprocessor.get_train_val_test_data()
    '''
    model_nb.train(X_train, y_train)
    model_nb.evaluate(X_test, y_test)

    plot_confusion_matrix(model_nb.get_model(), X_test, y_test, 'Naive Bayes')