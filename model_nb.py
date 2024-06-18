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
    

if __name__ == "__main__":
    
    '''from data import DataPreprocessor
    from utils import plot_confusion_matrix

    # Train the model on balanced data
    preprocessor = DataPreprocessor(file_path='dataset/dataset_1/new_balanced_data.csv')
    preprocessor.load_and_preprocess()
    preprocessor.split_data()
    preprocessor.oversample()
    X_train, _, _, y_train, _, _ = preprocessor.get_train_val_test_data()

    # Test the model on unbalanced data
    preprocessor = DataPreprocessor(test_file='dataset/dataset_1/unbalanced_test_data.csv')
    preprocessor.load_and_preprocess()
    X_test, y_test = preprocessor.get_test_data()'''

    from data_2 import X_train, y_train, X_test, y_test, X_test_balanced, y_test_balanced

    model_builder = NaiveBayesModelBuilder()
    model_builder.train(X_train, y_train)
    model_builder.evaluate(X_test, y_test)
    model_builder.evaluate(X_test_balanced, y_test_balanced)

    plot_confusion_matrix(model_builder.get_model(), X_test, y_test, 'Naive Bayes')
    plot_confusion_matrix(model_builder.get_model(), X_test_balanced, y_test_balanced, 'Naive Bayes')

