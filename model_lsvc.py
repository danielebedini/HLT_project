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

if __name__ == '__main__':
    from data import DataPreprocessor

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

    model_lsvc = LSVCModelBuilder()
    model_lsvc.train(X_train_balanced, y_train_balanced)
    model_lsvc.evaluate(X_val_unbalanced, y_val_unbalanced)
    #model_lsvc.evaluate(X_test_unbalanced, y_test_unbalanced)
    plot_confusion_matrix(model_lsvc.get_model(), X_val_unbalanced, y_val_unbalanced, 'Linear SVC')
    #plot_confusion_matrix(model_lsvc.get_model(), X_test_unbalanced, y_test_unbalanced, 'Linear SVC')

