import pandas as pd
from utils import preprocess_text_v2

"""
File that preprocesses the data from the second dataset.
The second dataset is already divided into training, validation, and test sets.
The datasets are in the dataset folder.
"""

def preprocess(X):
    """Preprocess text data."""
    X = X.apply(preprocess_text_v2)
    X = X.dropna()
    return X

# Load and preprocess the training data

file = 'dataset/training.csv'
train_data = pd.read_csv(file)

X = train_data['reviewText']
X_train = preprocess(X)

y_train = train_data['overall']

X_train = [str(text) for text in X_train]
# y_train = [int(label-1) for label in y_train] # only for DistilBERT

print("Training data preprocessed successfully!")

# Load and preprocess the validation set

file = 'dataset/validation.csv'
val_data = pd.read_csv(file)

X_val = val_data['reviewText']
y_val = val_data['overall']

X_val = preprocess(X_val)

X_val = [str(text) for text in X_val]
# y_val = [int(label-1) for label in y_val] # only for DistilBERT

# Load and preprocess the unbalanced test set

file = 'dataset/test_unbalanced.csv'
test_data = pd.read_csv(file)

X_test = test_data['reviewText']
y_test = test_data['overall']


X_test = preprocess(X_test)

X_test = [str(text) for text in X_test]
#y_test = [int(label-1) for label in y_test] # only for DistilBERT

# Load and preprocess the balanced test set

file = 'dataset/test_balanced.csv'
test_data = pd.read_csv(file)

X_test_balanced = test_data['reviewText']
y_test_balanced = test_data['overall']

X_test_balanced = preprocess(X_test_balanced)

X_test_balanced = [str(text) for text in X_test_balanced]
# y_test_balanced = [int(label-1) for label in y_test_balanced] # only for DistilBERT

# load and preprpcess the real test set

file = 'dataset/test_real_scenario.csv'
test_data = pd.read_csv(file)

X_test_real = test_data['reviewText']
y_test_real = test_data['overall']

X_test_real = preprocess(X_test_real)

X_test_real = [str(text) for text in X_test_real]
#y_test_real = [int(label-1) for label in y_test_real] # only for DistilBERT