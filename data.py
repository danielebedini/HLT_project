import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text_v2, oversampler
from sklearn.model_selection import train_test_split

"""
Class that preprocesses the data from a given dataset.
This class can also split the data into training, validation, and test sets.
It can also do oversampling of the training data.
"""

class DataPreprocessor:
    def __init__(self, file_path=None, train_file=None, val_file=None, test_file=None):
        self.file_path = file_path
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_and_preprocess(self):
        """Load and preprocess data."""
        if self.file_path:
            self.data = pd.read_csv(self.file_path)
            print("Data preprocessed successfully!")
            self.data['CleanedText'] = self.data['reviewText'].apply(preprocess_text_v2)
            self.data = self.data.dropna(subset=['CleanedText'])
            print("Preprocessing completed.")
        else:
            if self.train_file:
                self.X_train, self.y_train = self._load_and_preprocess_file(self.train_file)
            if self.val_file:
                self.X_val, self.y_val = self._load_and_preprocess_file(self.val_file)
            if self.test_file:
                self.X_test, self.y_test = self._load_and_preprocess_file(self.test_file)

    def _load_and_preprocess_file(self, file_path):
        """Load and preprocess a specific file."""
        data = pd.read_csv(file_path)
        data['CleanedText'] = data['reviewText'].apply(preprocess_text_v2)
        data = data.dropna(subset=['CleanedText'])
        X = data['CleanedText']
        y = data['overall']
        print(f"Data from {file_path} preprocessed successfully!")
        return X, y

    def split_data(self, test_size=0.25, validation_size=0.25, random_state=42, stratify_column='overall'):
        """Split data into training, validation, and test sets."""
        if not self.file_path:
            raise ValueError("File path must be provided for splitting data.")
        
        X = self.data['CleanedText']
        y = self.data[stratify_column]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Further split train set into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        print("Data split completed.")

    def get_train_data(self):
        """Get training data."""
        return self.X_train, self.y_train

    def get_val_data(self):
        """Get validation data."""
        return self.X_val, self.y_val

    def get_test_data(self):
        """Get test data."""
        return self.X_test, self.y_test

    def get_train_val_test_data(self):
        """Get split data."""
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def oversample(self):
        """Resample training data."""
        self.X_train, self.y_train = oversampler(self.X_train, self.y_train)
        print("Resampling completed.")