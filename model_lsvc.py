from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

from utils import plot_confusion_matrix

class LSVCModelBuilder:
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8):
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        self.model = Pipeline([
            ('tfidf', tfidf),
            ('scaler', StandardScaler(with_mean=False)),
            ('clf', LinearSVC(max_iter = 2000))
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
    
    from data import DataPreprocessor
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
    X_test, y_test = preprocessor.get_test_data()

    model_builder = LSVCModelBuilder()
    model_builder.train(X_train, y_train)
    model_builder.evaluate(X_test, y_test)

    plot_confusion_matrix(model_builder.get_model(), X_test, y_test, 'Count Vectorizer with Logistic Regression')

