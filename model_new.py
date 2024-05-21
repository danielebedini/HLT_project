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


if __name__ == '__main__':
    from data import DataPreprocessor
    from data_2 import X_train, X_test, y_train, y_test

    unbalanced_data = DataPreprocessor('unbalanced_test_data.csv')
    unbalanced_data.load_and_preprocess()
    unbalanced_data.split_data()
    unbalanced_data.oversample()
    _, _, X_test_unbalanced, _, _, y_test_unbalanced = unbalanced_data.get_train_val_test_data()

    '''
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
    '''

    model_builder = TfIdfLogisticRegressionModelBuilder()
    model_builder.train(X_train, y_train)
    model_builder.evaluate(X_test, y_test)
    #model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)

    plot_confusion_matrix(model_builder.get_model(), X_test, y_test, 'Logistic Regression')
    plot_confusion_matrix(model_builder.get_model(), X_test_unbalanced, y_test_unbalanced, 'Logistic Regression')
    #plot_confusion_matrix(model_builder.get_model(), X_test_unbalanced, y_test_unbalanced, 'Logistic Regression')