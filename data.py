import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text, preprocess_text_porter_stemmer, preprocess_text_v2, preprocess_text_contractions

# load dataset
#data = pd.read_csv('amazon_reviews.csv')
data_balanced = pd.read_csv('new_balanced_data.csv')
# Visualizza le prime righe del dataset
print(data_balanced.head())

# apply preprocessing to the column named 'reviewText'
data_balanced['CleanedText'] = data_balanced['reviewText'].apply(preprocess_text_v2)

# remove rows with missing values in the 'CleanedText' column
data_balanced = data_balanced.dropna(subset=['CleanedText'])

# visualize only a few columns
print(data_balanced[['CleanedText', 'overall']].head())

# divide the dataset into features and target
X = data_balanced['CleanedText']
y = data_balanced['overall']

# divide the dataset into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#X_train_balanced = data_balanced['CleanedText']
#y_train_balanced = data_balanced['overall']

# divide the balanced dataset into training and test sets

X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
