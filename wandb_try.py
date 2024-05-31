# This script needs these libraries to be installed:
#   numpy, sklearn
import wandb
import numpy as np
from wandb.sklearn import plot_precision_recall, plot_feature_importances
from wandb.sklearn import plot_class_proportions, plot_learning_curve, plot_roc

'''
# load and process data
wbcd = datasets.load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=test_size)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)
model_params = model.get_params()

# get predictions
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
'''
# Import amazon data
from data import X_train, X_val, X_test, y_train, y_val, y_test, X_train_unbalanced, X_val_unbalanced, X_test_unbalanced, y_train_unbalanced, y_val_unbalanced, y_test_unbalanced

# train model
from model_nb import NaiveBayesModelBuilder

model = NaiveBayesModelBuilder()
model.train(X_train, y_train)

# get predictions (on validation set)
y_pred = model.evaluate(X_val_unbalanced, y_val_unbalanced)


# start a new wandb run and add your model hyperparameters
wandb.init(project='hlt-project')

# Add additional configs to wandb
wandb.config.update({"test_size" : 0.2,
                    "train_len" : len(X_train),
                    "test_len" : len(X_test)})

# log additional visualisations to wandb
plot_class_proportions(y_train, y_test, y_pred)
plot_learning_curve(model, X_train, y_train)
#plot_roc(y_test)
plot_precision_recall(y_test, y_test, y_pred)
plot_feature_importances(model)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()