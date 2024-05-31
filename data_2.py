"""
- Use the whole balanced dataset for training.
- Use the whole unbalanced dataset for validation and testing.
"""
import pandas as pd
import matplotlib.pyplot as plt
from utils import preprocess_text_v2
from sklearn.model_selection import train_test_split

def preprocess(X):
    """Preprocess text data."""
    X = X.apply(preprocess_text_v2)
    X = X.dropna()
    return X

def plot_cake_graph(dataset, title):
    """Plot cake graph of a given dataset."""
    # Plot cake graph
    dataset['overall'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, title=title)
    plt.axis('equal')
    plt.show()

# Load and preprocess balanced training data

file = 'balanced_train_data.csv'
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

print("Training data preprocessed successfully!")

# Load and preprocess unbalanced test data

file = 'unbalanced_test_data.csv'
test_data = pd.read_csv(file)

# Estrai la colonna 'reviewText' e 'overall'
X = test_data['reviewText']
y = test_data['overall']

# Preprocessa i dati (assumendo che tu abbia una funzione preprocess)
X = preprocess(X)

# Dividi i dati in test e validation set con proporzione 50-50
X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=0.5, random_state=42)

X_train = X_train.values.tolist()
X_val = X_val.values.tolist()
X_test = X_test.values.tolist()

# Stampa i primi 3 elementi per verifica
#print(X_test[:3])
#print(y_test[:3])
#print(X_val[:3])
#print(y_val[:3])

# Stampa le forme dei nuovi set di dati
#print(X_test.shape)
#print(y_test.shape)
#print(X_val.shape)
#print(y_val.shape)

print("Testing data preprocessed successfully!")

if __name__ == '__main__':

    # Plot cake graph
    plot_cake_graph(train_data, "Balanced Training Data")   
    plot_cake_graph(test_data, "Unbalanced Test Data")