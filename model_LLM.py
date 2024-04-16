# Large language model for sentiment analysis

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import preprocess_text_v2

# Carica il dataset
data = pd.read_csv('amazon_reviews.csv')

# apply preprocessing to the column named 'reviewText'
data['CleanedText'] = data['reviewText'].apply(preprocess_text_v2)

# remove rows with missing values in the 'CleanedText' column
data = data.dropna(subset=['CleanedText'])

# visualize only a few columns
print(data[['CleanedText', 'overall']].head())

# divide the dataset into features and target
X = data['CleanedText']
y = data['overall']

# divide the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# divide the balanced dataset into training and test sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Large language model for sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# create a pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# train the model
model.fit(X_train_balanced, y_train_balanced)

# make predictions
y_pred = model.predict(X_test_balanced)

# calculate the accuracy
accuracy = accuracy_score(y_test_balanced, y_pred)
print(f'Accuracy: {accuracy:.2f}')

