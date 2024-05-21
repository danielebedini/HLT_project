from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from data import DataPreprocessor
from utils import plot_confusion_matrix

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Assicurati che le etichette siano interi e non one-hot, e sono scalari per ogni esempio.
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # dtype=torch.long per etichette di classificazione
        return item

class DistilBertModelBuilder:
    def __init__(self, num_labels, max_length=512):
        # Il tokenizer è definito come attributo della classe qui
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.max_length = max_length
        self.trainer = None

    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=8):
        # Utilizzo del tokenizer definito nell'oggetto
        train_dataset = SentimentDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = SentimentDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            report_to='wandb'
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda pred: {'accuracy': accuracy_score(y_val, np.argmax(pred.predictions, axis=1))}
        )

        self.trainer.train()

    def predict(self, X_test, y_test):
        test_dataset = SentimentDataset(X_test, y_test, self.tokenizer, self.max_length)
        predictions = self.trainer.predict(test_dataset)
        print(f"Predictions shape: {predictions.predictions.shape}")  # Dimensioni dell'output del modello
        print(f"Label shape: {predictions.label_ids.shape}")  # Dimensioni delle etichette
        y_pred = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'F1 score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred)) 

    def get_model(self):
        return self.model
    

if __name__ == '__main__':
    # load, preprocess and split balanced data
    from data import DataPreprocessor
    preprocessor = DataPreprocessor('new_balanced_data.csv')
    preprocessor.load_and_preprocess()
    preprocessor.split_data()
    #preprocessor.oversample()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.get_train_val_test_data()

    # convert X_train, X_val, X_test to list of strings
    X_train = [str(text) for text in X_train]
    X_val = [str(text) for text in X_val]
    X_test = [str(text) for text in X_test]

    # convert labels into list of integers
    y_train = [int(label-1) for label in y_train]
    y_val = [int(label-1) for label in y_val]
    y_test = [int(label-1) for label in y_test]

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    model = DistilBertModelBuilder(num_labels=5)

    model.train(X_train, y_train, X_val, y_val, epochs=3, batch_size=8)

    model.predict(X_test, y_test)
