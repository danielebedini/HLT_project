from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils import metrics_with_three_classes

class LSVCModelBuilder:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('clf', SGDClassifier()),
            ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test): # It can take even validation data
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.model

if __name__ == "__main__":
    
    """from data import DataPreprocessor
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
    X_test, y_test = preprocessor.get_test_data()"""

    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced

    # Linear SVC model
    model_lsvc = LSVCModelBuilder()
    model_lsvc.train(X_train, y_train)
    metrics_with_three_classes(model_lsvc.model, X_test, y_test, 'Linear SVC Unbalanced Data')
    metrics_with_three_classes(model_lsvc.model, X_test_balanced, y_test_balanced, 'Linear SVC Balanced Data')

