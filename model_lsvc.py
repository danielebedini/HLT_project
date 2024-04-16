from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

class TextModelBuilder:
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8):
        tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, max_df=max_df)
        self.model = Pipeline([
            ('tfidf', tfidf),
            ('clf', LinearSVC())
        ])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.model
