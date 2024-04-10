import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text, preprocess_text_porter_stemmer, preprocess_text_v2, preprocess_text_contractions

# Carica il dataset
data = pd.read_csv('amazon_reviews.csv')

# Visualizza le prime righe del dataset
print(data.head())

# apply preprocessing to the column named 'reviewText'
data['CleanedText'] = data['reviewText'].apply(preprocess_text)
#data['CleanedText'] = data['reviewText'].apply(preprocess_text_porter_stemmer)
#data['CleanedText'] = data['reviewText'].apply(preprocess_text_v2)
#data['CleanedText'] = data['reviewText'].apply(preprocess_text_contractions)


# remove rows with missing values in the 'CleanedText' column
data = data.dropna(subset=['CleanedText'])

# visualize only a few columns
print(data[['CleanedText', 'overall']].head())

# divide the dataset into features and target
X = data['CleanedText']
y = data['overall']

# divide the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)