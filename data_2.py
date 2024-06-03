import pandas as pd
from utils import preprocess_text_v2
from sklearn.model_selection import train_test_split

def preprocess(X):
    """Preprocess text data."""
    X = X.apply(preprocess_text_v2)
    X = X.dropna()
    return X

# Load and preprocess balanced training data

file = 'dataset/training.csv'
train_data = pd.read_csv(file)
#print(data.head())

# List of string of the reviewText column
X = train_data['reviewText']
X_train = preprocess(X)
#print(X[:3])

# List of integers of the overall column
y_train = train_data['overall']
#print(y_train[:3])

#print(X_train.shape)
#print(y_train.shape)

#X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

X_train = [str(text) for text in X_train]
y_train = [int(label-1) for label in y_train]

print("Training data preprocessed successfully!")

file = 'dataset/validation.csv'
val_data = pd.read_csv(file)

# Estrai la colonna 'reviewText' e 'overall'
X_val = val_data['reviewText']
y_val = val_data['overall']

# Preprocessa i dati (assumendo che tu abbia una funzione preprocess)
X_val = preprocess(X_val)

X_val = [str(text) for text in X_val]
y_val = [int(label-1) for label in y_val]

# Load and preprocess unbalanced test data

file = 'dataset/test_imbalanced.csv'
test_data = pd.read_csv(file)

# Estrai la colonna 'reviewText' e 'overall'
X_test = test_data['reviewText']
y_test = test_data['overall']

# Preprocessa i dati (assumendo che tu abbia una funzione preprocess)
X_test = preprocess(X_test)

X_test = [str(text) for text in X_test]
y_test = [int(label-1) for label in y_test]

file = 'dataset/test_balanced.csv'
test_data = pd.read_csv(file)

# Estrai la colonna 'reviewText' e 'overall'
X_test_balanced = test_data['reviewText']
y_test_balanced = test_data['overall']

# Preprocessa i dati (assumendo che tu abbia una funzione preprocess)
X_test_balanced = preprocess(X_test_balanced)

X_test_balanced = [str(text) for text in X_test_balanced]
y_test_balanced = [int(label-1) for label in y_test_balanced]
