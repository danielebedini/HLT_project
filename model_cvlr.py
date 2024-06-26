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

    model_cvlr = LogisticRegressionModelBuilder()
    model_cvlr.train(X_train, y_train)

    
    """print("Test (difficult version) on balanced data")
    model_cvlr.evaluate(X_test_balanced, y_test_balanced)
    plot_confusion_matrix(model_cvlr.model, X_test_balanced, y_test_balanced, 'Logistic Regression Balanced Data')

    print("Test (difficult version) on unbalanced data")
    model_cvlr.evaluate(X_test, y_test)
    plot_confusion_matrix(model_cvlr.model, X_test, y_test, 'Logistic Regression Unbalanced Data')"""

    """print("Test (difficult version) on real data")
    model_cvlr.evaluate(X_test_real, y_test_real)
    plot_confusion_matrix(model_cvlr.model, X_test_real, y_test_real, 'Logistic Regression Real Test')"""
    

    print("Test on balanced data")
    metrics_with_three_classes(model_cvlr.model, X_test_balanced, y_test_balanced, 'Count Vectorizer Logistic Regression Balanced Data')
    
    #print("Test on unbalanced data")
    #metrics_with_three_classes(model_cvlr.model, X_test, y_test, 'Naive Bayes Unbalanced Data')

    print("Test on real data")
    metrics_with_three_classes(model_cvlr.model, X_test_real, y_test_real, 'Count Vectorizer Logistic Regression Real Test')