import pandas as pd
from sklearn.model_selection import train_test_split
from utils import preprocess_text_v2, preprocess_text_contractions, get_wordcloud, oversampler

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess(self):
        """Carica e preprocessa i dati."""
        self.data = pd.read_csv(self.file_path)
        print("Dati caricati con successo.")
        self.data['CleanedText'] = self.data['reviewText'].apply(preprocess_text_v2)
        self.data = self.data.dropna(subset=['CleanedText'])
        print("Preprocessamento completato.")

    def split_data(self, test_size=0.2, random_state=42, stratify_column='overall'):
        """Divide i dati in set di training e test."""
        X = self.data['CleanedText']
        y = self.data[stratify_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("Divisione in training e test set completata.")

    def get_train_test_data(self):
        """Ritorna i dati di training e test."""
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def oversample(self):
        """Esegue il resampling dei dati."""
        self.X_train, self.y_train = oversampler(self.X_train, self.y_train)
        print("Resampling completato.")
