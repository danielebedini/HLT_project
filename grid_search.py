from sklearn.model_selection import GridSearchCV
from data import DataPreprocessor
from utils import metrics_with_three_classes

""" 
This class is used to optimize the hyperparameters of a given model using GridSearchCV, for model selection. 
It evaluates the model before and after optimization for more accurate results.
"""

class ModelOptimizer:
    def __init__(self, model, param_grid):
        # Initialize GridSearchCV with the given model and parameter grid
        self.grid_search = GridSearchCV(model, param_grid, cv=4, scoring='f1_weighted') # cv=4 is the number of folds
        self.best_params = None
        # self.classification_report = None

    def fit(self, X_train, y_train):
        # Fit the model using GridSearchCV
        self.grid_search.fit(X_train, y_train)
        self.best_params = self.grid_search.best_params_
        print("Best Parameters:", self.best_params)

    def evaluate(self, X_test, y_test, model_name='Model'):
        # Predict and evaluate the model on the test data
        metrics_with_three_classes(self.grid_search, X_test, y_test, model_name)

    def get_model(self):
        # Return the best estimator found by GridSearchCV
        return self.grid_search.best_estimator_


if __name__ == '__main__':

    from model_lsvc import LSVCModelBuilder
    from model_cvlr import LogisticRegressionModelBuilder
    from model_tflr import TfIdfLogisticRegressionModelBuilder
    from model_nb import NaiveBayesModelBuilder

    from data_2 import X_train, X_test_real, y_train, y_test_real, X_test_balanced, y_test_balanced # these data are already preprocessed and ready to be used

    # We are going to optimize the hyperparameters of all the models
    # uncomment the model you want to optimize

    # The methodology is the same for all the models, we are going to use GridSearchCV to find the best hyperparameters, then 
    # we will see the classification report and the confusion matrix of the model before and after optimization
    
    # Naive Bayes model
    model = NaiveBayesModelBuilder().get_model()
    param_grid = {
        'clf__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'clf__fit_prior': [True, False],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': [True, False],
        'tfidf__smooth_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__norm': ['l1', 'l2', None]
    }

    nb = NaiveBayesModelBuilder()
    nb.train(X_train, y_train)
    metrics_with_three_classes(nb.model, X_test_real, y_test_real, 'Naive Bayes Unbalanced Data')

    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train, y_train)
    optimizer.evaluate(X_test_real, y_test_real, 'Naive Bayes Unbalanced Data Optimized')


    # Logistic Regression model
    model = LogisticRegressionModelBuilder().get_model()

    param_grid = {
        'classifier__C': [0.1, 0.5, 1.0, 1.5, 2.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__max_iter': [100, 200, 500],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vectorizer__use_idf': [True, False],
        'vectorizer__smooth_idf': [True, False],
        'vectorizer__sublinear_tf': [True, False],
        'vectorizer__norm': ['l1', 'l2', None]
    }

    cvlr = LogisticRegressionModelBuilder()
    cvlr.train(X_train, y_train)
    metrics_with_three_classes(cvlr.model, X_test_real, y_test_real, 'Count Vectorizer Logistic Regression Unbalanced Data')

    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train, y_train)
    optimizer.evaluate(X_test_real, y_test_real, 'Count Vectorizer Logistic Regression Unbalanced Data Optimized')

    # Linear Support Vector Classification model
    model = LSVCModelBuilder()

    param_grid = {
        'clf__alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': [True, False],
        'tfidf__smooth_idf': [True, False],
        'tfidf__sublinear_tf': [True, False],
        'tfidf__norm': ['l1', 'l2', None]
    }

    lsvc = LSVCModelBuilder()
    lsvc.train(X_train, y_train)
    metrics_with_three_classes(lsvc.model, X_test_real, y_test_real, 'LSVC Unbalanced Data')

    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train, y_train)
    optimizer.evaluate(X_test_real, y_test_real, 'LSVC Unbalanced Data Optimized')


    # TfIdf Logistic Regression model
    model = TfIdfLogisticRegressionModelBuilder()

    param_grid = {
        'classifier__C': [0.1, 0.5, 1.0, 1.5, 2.0],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__max_iter': [100, 200, 500],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'vectorizer__use_idf': [True, False],
        'vectorizer__smooth_idf': [True, False],
        'vectorizer__sublinear_tf': [True, False],
        'vectorizer__norm': ['l1', 'l2', None]
    }

    tflr = TfIdfLogisticRegressionModelBuilder()
    tflr.train(X_train, y_train)
    metrics_with_three_classes(tflr.model, X_test_real, y_test_real, 'TfIdf Logistic Regression Unbalanced Data')

    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train, y_train)

    optimizer.evaluate(X_test_real, y_test_real, 'TfIdf Logistic Regression Unbalanced Data Optimized')

    