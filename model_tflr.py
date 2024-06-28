from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from scipy.sparse import issparse
from utils import plot_confusion_matrix, metrics_with_three_classes

# As suggested, we use TfidfVectorizer instead of CountVectorizer with logistic regression

class TfIdfLogisticRegressionModelBuilder:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(1, 2))),
            ('classifier', LogisticRegression(max_iter=500))
        ])

    def train(self, X_train, y_train):

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


if __name__ == '__main__':

    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced, X_test_real, y_test_real
    from utils import plot_confusion_matrix

    model_tflr = TfIdfLogisticRegressionModelBuilder()
    model_tflr.train(X_train, y_train)

    
    """print("Test (difficult version) on balanced data")
    model_tflr.evaluate(X_test_balanced, y_test_balanced)
    plot_confusion_matrix(model_tflr.model, X_test_balanced, y_test_balanced, 'TfIdf Logistic Regression Balanced Data')

    print("Test (difficult version) on unbalanced data")
    model_tflr.evaluate(X_test, y_test)
    plot_confusion_matrix(model_tflr.model, X_test, y_test, 'TfIdf Logistic Regression Unbalanced Data')
"""
    """print("Test (difficult version) on unbalanced data")
    model_tflr.evaluate(X_test_real, y_test_real)
    plot_confusion_matrix(model_tflr.model, X_test_real, y_test_real, 'TfIdf Logistic Regression Real Test')
    """

    #print("Test on balanced data")
    #metrics_with_three_classes(model_tflr.model, X_test_balanced, y_test_balanced, 'TF-IDF Logistic Regression Real Test')

    #print("Test on unbalanced data")
    #metrics_with_three_classes(model_tflr.model, X_test, y_test, 'TF-IDF Logistic Regression Real Test')

    print("Test on real data")
    metrics_with_three_classes(model_tflr.model, X_test_real, y_test_real, 'TF-IDF Logistic Regression Real Test')