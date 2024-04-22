from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data import DataPreprocessor
from model_lsvc import TextModelBuilder
from model_rfc import RandomForestModelBuilder
from model_LLM import LogisticRegressionModelBuilder
from model_new import TfIdfLogisticRegressionModelBuilder
from utils import save_results_json, plot_confusion_matrix

class ModelOptimizer:
    def __init__(self, model, param_grid):
        self.grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        self.best_params = None
        self.classification_report = None
        self.accuracy = None

    def fit(self, X_train, y_train):
        self.grid_search.fit(X_train, y_train)
        self.best_params = self.grid_search.best_params_
        print("Best Parameters:", self.best_params)

    def evaluate(self, X_test, y_test):
        y_pred = self.grid_search.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy after Grid Search: {self.accuracy}')
        self.classification_report = classification_report(y_test, y_pred,zero_division=0)
        print(f'Classification Report:\n{self.classification_report}')

    def save_results(self, filename):
        save_results_json(self.grid_search, self.accuracy, self.classification_report, filename)

# load, preprocess and split unbalanced data
unbalanced_data = DataPreprocessor('amazon_reviews.csv')
unbalanced_data.load_and_preprocess()
unbalanced_data.split_data()
X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced = unbalanced_data.get_train_test_data()

# load, preprocess and split balanced data
preprocessor = DataPreprocessor('new_balanced_data.csv')
preprocessor.load_and_preprocess()
preprocessor.split_data()
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = preprocessor.get_train_test_data()


print("Balanced Data:")
print(f"X_train_balanced: {X_train_balanced.head()}")
print(f"X_test_balanced: {X_test_balanced.head()}")

'''
# Creazione e allenamento del modello
model_builder = TextModelBuilder()
model_builder.train(X_train_balanced, y_train_balanced)
model_builder.evaluate(X_test_balanced, y_test_balanced)

# Ottimizzazione del modello
model = model_builder.get_model()
param_grid = {'clf__C': [0.1, 1.0, 10.0, 100.0], 'clf__loss': ['hinge', 'squared_hinge']}
optimizer = ModelOptimizer(model, param_grid)
optimizer.fit(X_train_balanced, y_train_balanced)
optimizer.evaluate(X_test_balanced, y_test_balanced)
'''

'''
lr_model_builder = LogisticRegressionModelBuilder(max_iter=5000, solver='liblinear')  
lr_model_builder.train(X_train_balanced, y_train_balanced)
lr_model_builder.evaluate(X_test_balanced, y_test_balanced)

# Further optimization
model = lr_model_builder.get_model()
param_grid = {
    'classifier__C': [0.1, 1.0],
#    'vectorizer__max_features': [None, 5000, 10000],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
#    'vectorizer__min_df': [5, 10],
#    'vectorizer__max_df': [0.5,  1.0],
#    'classifier__solver': ['liblinear']  
}
optimizer = ModelOptimizer(model, param_grid)
optimizer.fit(X_train_balanced, y_train_balanced)
optimizer.evaluate(X_test_balanced, y_test_balanced)
'''

'''
#rf_model_builder = RandomForestModelBuilder()
#rf_model_builder.train(X_train_balanced, y_train_balanced)
#rf_model_builder.evaluate(X_test_balanced, y_test_balanced)

# Ottimizzazione del modello
#model = rf_model_builder.get_model()
param_grid = {
    'clf__n_estimators': [100],
    'clf__max_depth': [None, 10],
    'clf__min_samples_split': [2],
    'clf__min_samples_leaf': [1, 2, 4],
    'tfidf__max_features': [None, 5000],
    'tfidf__ngram_range': [(1, 1)],
    'tfidf__min_df': [1, 5],
    'tfidf__max_df': [0.50]
}
#optimizer = ModelOptimizer(model, param_grid)
#optimizer.fit(X_train_balanced, y_train_balanced)
#optimizer.evaluate(X_test_balanced, y_test_balanced)
'''

lr_model_builder = LogisticRegressionModelBuilder(max_iter=5000)
lr_model_builder.train(X_train_balanced, y_train_balanced)
lr_model_builder.evaluate(X_test_balanced, y_test_balanced)

# Further optimization
model = lr_model_builder.get_model()
param_grid = {
    'classifier__C': [0.1, 1.0],
    #'vectorizer__max_features': [None, 5000, 10000],
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    #'vectorizer__min_df': [5, 10],
    #'vectorizer__max_df': [0.5,  1.0],
    'classifier__solver': ['liblinear']
}
optimizer = ModelOptimizer(model, param_grid)
optimizer.fit(X_train_balanced, y_train_balanced)
optimizer.evaluate(X_test_unbalanced, y_test_unbalanced)

plot_confusion_matrix(optimizer.grid_search, X_test_unbalanced, y_test_unbalanced)