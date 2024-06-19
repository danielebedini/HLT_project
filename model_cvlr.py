from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import issparse
from utils import plot_confusion_matrix, metrics_with_three_classes

""" This pipeline uses CountVectorizer with Logistic Regression. """

class LogisticRegressionModelBuilder:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', LogisticRegression(max_iter=500))
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
    
    """from data import DataPreprocessor

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

    # Logistic regression model (with count vectorizer)
    model_cvlr = LogisticRegressionModelBuilder()
    model_cvlr.train(X_train, y_train)
    metrics_with_three_classes(model_cvlr.model, X_test, y_test, 'CVLR Unbalanced Data')
    metrics_with_three_classes(model_cvlr.model, X_test_balanced, y_test_balanced, 'CVLR Balanced Data')