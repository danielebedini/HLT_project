import pandas as pd

class DataProcessor:
    def __init__(self, input_file, output_file, down_sample_size=300, random_state=42):
        self.input_file = input_file
        self.output_file = output_file
        self.down_sample_size = down_sample_size
        self.random_state = random_state
        self.data = None
        self.five_star_reviews = None
        self.not_five_star_reviews = None
        self.final_data = None

    def load_data(self):
        """Carica i dati dal file CSV."""
        self.data = pd.read_csv(self.input_file)
        print("Dati caricati con successo.")

    def filter_data(self, rating=5.0):
        """Filtra le recensioni con 5 stelle e esegui il down sampling."""
        self.five_star_reviews = self.data[self.data['overall'] == rating]
        self.not_five_star_reviews = self.data[self.data['overall'] != rating]
        print("Recensioni filtrate con successo.")

    def down_sample_data(self):
        """Down-sampling delle recensioni a cinque stelle."""
        down_sampled = self.five_star_reviews.sample(n=self.down_sample_size, random_state=self.random_state)
        self.final_data = pd.concat([self.not_five_star_reviews, down_sampled])
        print("Down-sampling completato con successo.")

    def save_data(self):
        """Salva il dataset finale in un nuovo file CSV."""
        self.final_data.to_csv(self.output_file, index=False)
        print(f"Dati salvati con successo in {self.output_file}.")

# Uso della classe
processor = DataProcessor('amazon_reviews.csv', 'new_balanced_data.csv')
processor.load_data()
processor.filter_data()
processor.down_sample_data()
processor.save_data()
