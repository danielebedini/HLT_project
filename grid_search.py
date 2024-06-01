from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data import DataPreprocessor
from model_lsvc import LSVCModelBuilder
from model_rfc import RandomForestModelBuilder
from model_LLM import LogisticRegressionModelBuilder
from model_new import TfIdfLogisticRegressionModelBuilder
from model_nb import NaiveBayesModelBuilder
from utils import save_results_json, plot_confusion_matrix

class ModelOptimizer:
    def __init__(self, model, param_grid):
        # Initialize GridSearchCV with the given model and parameter grid
        self.grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        self.best_params = None
        self.classification_report = None
        self.accuracy = None

    def fit(self, X_train, y_train):
        # Fit the model using GridSearchCV
        self.grid_search.fit(X_train, y_train)
        self.best_params = self.grid_search.best_params_
        print("Best Parameters:", self.best_params)

    def evaluate(self, X_test, y_test):
        # Predict and evaluate the model on the test data
        y_pred = self.grid_search.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy after Grid Search: {self.accuracy}')
        self.classification_report = classification_report(y_test, y_pred, zero_division=0)
        print(f'Classification Report:\n{self.classification_report}')

    def save_results(self, filename):
        # Save the results to a JSON file
        save_results_json(self.grid_search, self.accuracy, self.classification_report, filename)

    def get_model(self):
        # Return the best estimator found by GridSearchCV
        return self.grid_search.best_estimator_

if __name__ == '__main__':
    # Load, preprocess, and oversample the data
    preprocessor = DataPreprocessor(train_file='dataset/training.csv', val_file='dataset/validation.csv', test_file='dataset/test_balanced.csv')
    preprocessor.load_and_preprocess()
    preprocessor.oversample()
    X_train_balanced, X_val_balanced, X_test_balanced, y_train_balanced, y_val_balanced, y_test_balanced = preprocessor.get_train_val_test_data()

    preprocessor = DataPreprocessor(test_file='dataset/test_unbalanced.csv')
    preprocessor.load_and_preprocess()
    X_test_unbalanced, y_test_unbalanced = preprocessor.get_test_data()

    print("Balanced data:")
    print("TR balanced: ", len(X_train_balanced))
    print("VL balanced: ", len(X_val_balanced))
    print("TS balanced: ", len(X_test_balanced))
    print("TS unbalanced: ", len(X_test_unbalanced))




    # Linear SVC model --------------------------------------------------------
    print("\n\nLinear SVC model")
    model_builder = LSVCModelBuilder()
    model_builder.train(X_train_balanced, y_train_balanced)
    model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)
    # Further optimization for Linear SVC
    model = model_builder.get_model()
    param_grid = {
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__loss': ['hinge', 'squared_hinge'],
        'clf__class_weight': [None, 'balanced']
    }
    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train_balanced, y_train_balanced)
    optimizer.evaluate(X_test_unbalanced, y_test_unbalanced)
    # plot confusion matrix
    plot_confusion_matrix(optimizer.get_model(), X_test_unbalanced, y_test_unbalanced, model_name="Linear SVC")
    plot_confusion_matrix(optimizer.get_model(), X_test_balanced, y_test_balanced, model_name="Linear SVC balanced")



    # Logistic regression model (with count vectorizer) -----------------------
    print("\n\nLogistic regression model")
    lr_model_builder = LogisticRegressionModelBuilder(max_iter=5000, solver='liblinear')
    lr_model_builder.train(X_train_balanced, y_train_balanced)
    lr_model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)
    # Further optimization for Logistic Regression
    model = lr_model_builder.get_model()
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'vectorizer__max_features': [None, 5000, 10000, 20000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__min_df': [1, 5, 10],
        'vectorizer__max_df': [0.5, 0.75, 1.0]
    }
    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train_balanced, y_train_balanced)
    optimizer.evaluate(X_test_unbalanced, y_test_unbalanced)
    # plot confusion matrix
    plot_confusion_matrix(optimizer.get_model(), X_test_unbalanced, y_test_unbalanced, model_name="Logistic Regression")
    plot_confusion_matrix(optimizer.get_model(), X_test_balanced, y_test_balanced, model_name="Logistic Regression balanced")



    # TfIdf Logistic Regression model -----------------------------------------
    print("\n\nTfIdfLogisticRegressionModelBuilder")
    tfidf_lr_model_builder = TfIdfLogisticRegressionModelBuilder()
    tfidf_lr_model_builder.train(X_train_balanced, y_train_balanced)
    tfidf_lr_model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)
    # Further optimization for TfIdf Logistic Regression
    model = tfidf_lr_model_builder.get_model()
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__max_df': [0.5, 0.75, 1.0],
        'classifier__solver': ['liblinear', 'saga']
    }
    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train_balanced, y_train_balanced)
    optimizer.evaluate(X_test_unbalanced, y_test_unbalanced)
    # plot confusion matrix
    plot_confusion_matrix(optimizer.get_model(), X_test_unbalanced, y_test_unbalanced, model_name="TF-IDF Logistic Regression")
    plot_confusion_matrix(optimizer.get_model(), X_test_balanced, y_test_balanced, model_name="TF-IDF Logistic Regression balanced")

    # Naive Bayes model
    print("\n\nNaive Bayes model")
    nb_model_builder = NaiveBayesModelBuilder()
    nb_model_builder.train(X_train_balanced, y_train_balanced)
    nb_model_builder.evaluate(X_test_unbalanced, y_test_unbalanced)
    # Further optimization for Naive Bayes
    model = nb_model_builder.get_model()
    param_grid = {
        'clf__alpha': [0.01, 0.1, 1, 10],
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [1, 5, 10],
        'tfidf__max_df': [0.5, 0.75, 1.0],
        'tfidf__stop_words': [None, 'english']
    }
    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train_balanced, y_train_balanced)
    optimizer.evaluate(X_test_unbalanced, y_test_unbalanced)
    # plot confusion matrix
    plot_confusion_matrix(optimizer.get_model(), X_test_unbalanced, y_test_unbalanced, model_name="Naive Bayes")
    plot_confusion_matrix(optimizer.get_model(), X_test_balanced, y_test_balanced, model_name="Naive Bayes balanced")
