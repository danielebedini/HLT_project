import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text_v2, preprocess_text_contractions, get_wordcloud, oversampler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_and_preprocess(self):
        """Load and preprocess data."""
        self.data = pd.read_csv(self.file_path)
        print("Data preprocessed successfully!")
        self.data['CleanedText'] = self.data['reviewText'].apply(preprocess_text_v2)
        self.data = self.data.dropna(subset=['CleanedText'])
        print("Preprocessing completed.")

    def split_data(self, test_size=0.2, validation_size=0.25, random_state=42, stratify_column='overall'):
        """Split data into training, validation, and test sets."""
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

    def get_train_val_test_data(self):
        """Get split data."""
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def oversample(self):
        """Resample training data."""
        self.X_train, self.y_train = oversampler(self.X_train, self.y_train)
        print("Resampling completed.")

preprocessor = DataPreprocessor('balanced_train_data.csv')
preprocessor.load_and_preprocess()
preprocessor.split_data()
preprocessor.oversample()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_train_val_test_data()

unbalanced_data = DataPreprocessor('unbalanced_test_data.csv')
unbalanced_data.load_and_preprocess()
unbalanced_data.split_data()
unbalanced_data.oversample()
X_train_unbalanced, X_val_unbalanced, X_test_unbalanced, y_train_unbalanced, y_val_unbalanced, y_test_unbalanced = unbalanced_data.get_train_val_test_data()

if __name__ == '__main__':
    print("TRAIN unbalanced data: ", len(X_train))
    print("VAL unbalanced data: ", len(X_val))
    print("TEST unbalanced data: ", len(X_test))

    print("TRAIN balanced data: ", len(X_train_unbalanced))
    print("VAL balanced data: ", len(X_val_unbalanced))
    print("TEST balanced data: ", len(X_test_unbalanced))