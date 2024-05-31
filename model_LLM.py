from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import issparse

class LogisticRegressionModelBuilder:
    def __init__(self, max_iter=20000, solver='liblinear'):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
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
    #from data import X_train, X_test_unbalanced, y_train, y_test_unbalanced, X_val_unbalanced, y_val_unbalanced
    from data_2 import X_train, X_test, y_train, y_test
    from utils import plot_confusion_matrix
    model_builder = LogisticRegressionModelBuilder()
    model_builder.train(X_train, y_train)
    model_builder.evaluate(X_test, y_test)
    #model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)

    plot_confusion_matrix(model_builder.get_model(), X_test, y_test, 'Count Vectorizer with Logistic Regression')
    #plot_confusion_matrix(model_builder.get_model(), X_test_unbalanced, y_test_unbalanced, 'Logistic Regression')