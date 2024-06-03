from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import issparse
from utils import plot_confusion_matrix

# As suggested, we use TfidfVectorizer instead of CountVectorizer with logistic regression

class TfIdfLogisticRegressionModelBuilder:
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8, max_iter=20000, solver='liblinear'):
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        self.model = Pipeline([
            ('vectorizer', tfidf),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', LogisticRegression(max_iter=max_iter, solver=solver, verbose=0))
        ])

    def train(self, X_train, y_train):
        # Verifica se i dati sono sparsi e imposta with_mean di conseguenza
        X_transformed = self.model.named_steps['vectorizer'].fit_transform(X_train)
        if not issparse(X_transformed):
            self.model.named_steps['scaler'].set_params(with_mean=True)

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

    model_builder = TfIdfLogisticRegressionModelBuilder()
    model_builder.train(X_train, y_train)
    model_builder.evaluate(X_test, y_test)

    plot_confusion_matrix(model_builder.get_model(), X_test, y_test, 'Count Vectorizer with Logistic Regression')

