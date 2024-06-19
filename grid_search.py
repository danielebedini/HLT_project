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

    from data_2 import X_train, X_test, y_train, y_test # these data are already preprocessed and ready to be used

    # We are going to optimize the hyperparameters of all the models
    # uncomment the model you want to optimize

    # The methodology is the same for all the models, we are going to use GridSearchCV to find the best hyperparameters, then 
    # we will see the classification report and the confusion matrix of the model before and after optimization

    # Linear SVC model
    # model = LSVCModelBuilder().get_model()
    # param_grid = {
    #     'clf__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    #     'clf__penalty': ['l1', 'l2', 'elasticnet'],
    #     'clf__alpha': [0.0001, 0.001, 0.01],
    #     'clf__max_iter': [1000, 2000, 3000]
    # }
    
    # optimizer = ModelOptimizer(model, param_grid)
    # optimizer.fit(X_train, y_train)
    # optimizer.evaluate(X_test, y_test, 'Linear SVC Unbalanced Data')

    # CountVectorizer with Logistic Regression model
    # model = LogisticRegressionModelBuilder().get_model()
    # param_grid = {
    #     'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    #     'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    #     'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    # }

    # optimizer = ModelOptimizer(model, param_grid)
    # optimizer.fit(X_train, y_train)
    # optimizer.evaluate(X_test, y_test, 'CountVectorizer with Logistic Regression Unbalanced Data')

    # TfIdfVectorizer with Logistic Regression model
    #model = TfIdfLogisticRegressionModelBuilder().get_model()
    #param_grid = {
    #    'classifier__penalty': ['l1', 'l2', 'elasticnet'],
    #    'classifier__C': [0.001, 0.01, 0.1, 1, 10],
    #    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    #}

    #optimizer = ModelOptimizer(model, param_grid)
    #optimizer.fit(X_train, y_train)
    #optimizer.evaluate(X_test, y_test, 'TfIdfVectorizer with Logistic Regression Unbalanced Data')
    
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
    metrics_with_three_classes(nb.model, X_test, y_test, 'Naive Bayes Unbalanced Data')

    optimizer = ModelOptimizer(model, param_grid)
    optimizer.fit(X_train, y_train)
    optimizer.evaluate(X_test, y_test, 'Naive Bayes Unbalanced Data Optimized')
