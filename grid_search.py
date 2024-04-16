from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from data import X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced
from model_lsvc import model_lsvc
from model_rfc import model_rfc
from utils import save_results_json

K_FOLD = 5

# define the range of hyperparameters to test
lsvc_param_grid = {
    'clf__C': [0.1, 1.0, 10.0, 100.0],  # different values for C
    'clf__loss': ['hinge', 'squared_hinge']  # loss functions
}

rfc_param_grid = {
    'tfidf__max_features': [10000, 20000, 30000],  # different values for max_features
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # different values for ngram_range
    'tfidf__min_df': [5, 10],  # different values for min_df
    'tfidf__max_df': [0.7, 0.8, 0.9]  # different values for max_df
}

llm_param_grid = {
    'vectorizer__max_features': [10000, 20000, 30000],  # different values for max_features
    'vectorizer__ngram_range': [(1, 1), (1, 2)],  # different values for ngram_range
    'vectorizer__min_df': [5, 10],  # different values for min_df
    'vectorizer__max_df': [0.7, 0.8, 0.9],  # different values for max_df
    'classifier__C': [0.1, 1.0, 10.0, 100.0],  # different values for C
    'classifier__loss': ['hinge', 'squared_hinge']  # loss functions
}

# perform the grid search for the llm
#grid_search = GridSearchCV(model, llm_param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train_balanced, y_train_balanced)

# for linear SVC
grid_search = GridSearchCV(model_lsvc, lsvc_param_grid, cv=K_FOLD, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# For random forest
#grid_search = GridSearchCV(model_rfc, rfc_param_grid, cv=5, scoring='accuracy')
#grid_search.fit(X_train, y_train)

# best parameters found by GridSearchCV
print("Best Parameters:", grid_search.best_params_)

# evaluate the model on the test set
y_pred = grid_search.predict(X_test_balanced)
accuracy = accuracy_score(y_test_balanced, y_pred)
print(f'Accuracy after Grid Search: {accuracy}')

# Results:
class_report = classification_report(y_test_balanced, y_pred)
print(f'Classification Report:\n{class_report}')

# for linear SVC
# Best Parameters: {'clf__C': 1.0, 'clf__loss': 'hinge'}
# Accuracy after Grid Search: 0.8189216683621566

# For random forest
# results: Best Parameters: {'tfidf__max_df': 0.7, 'tfidf__max_features': 10000, 'tfidf__min_df': 5, 'tfidf__ngram_range': (1, 1)}
# Accuracy after Grid Search: 0.7985757884028484
save_results_json(grid_search, accuracy, class_report, 'grid_search_results.json')