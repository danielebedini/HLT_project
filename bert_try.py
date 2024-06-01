import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # in MB

class BertModelBuilder:
    def __init__(self, num_labels, max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.max_length = max_length
        self.trainer = None

    def train(self, X_train, y_train, X_val, y_val, epochs=3, batch_size=4):
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
            log_level='info',  # Set logging level
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda pred: {'accuracy': accuracy_score(y_val, np.argmax(pred.predictions, axis=1))}
        )

        logger.info("Starting training")
        logger.info(f"Memory usage before training: {memory_usage()} MB")
        self.trainer.train()
        logger.info(f"Memory usage after training: {memory_usage()} MB")
        logger.info("Training completed")

    def predict(self, X_test, y_test):
        test_dataset = SentimentDataset(X_test, y_test, self.tokenizer, self.max_length)
        predictions = self.trainer.predict(test_dataset)
        print(f"Predictions shape: {predictions.predictions.shape}")
        print(f"Label shape: {predictions.label_ids.shape}")
        y_pred = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f'F1 score: {f1:.2f}')
        print(f'Accuracy: {accuracy:.2f}')
        print(classification_report(y_test, y_pred))

    def get_model(self):
        return self.model


if __name__ == '__main__':
    from data import DataPreprocessor

    data = DataPreprocessor('amazon_reviews.csv')
    data.load_and_preprocess()
    data.split_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_train_val_test_data()

    # convert X_train, X_val, X_test to list of strings
    X_train = [str(text) for text in X_train]
    X_val = [str(text) for text in X_val]
    X_test = [str(text) for text in X_test]

    # convert labels into list of integers
    y_train = [int(label-1) for label in y_train]
    y_val = [int(label-1) for label in y_val]
    y_test = [int(label-1) for label in y_test]

    # print how many reviews with 1 star are in the training set
    print("1 star reviews in training set: ", y_train.count(0))
    print("2 star reviews in training set: ", y_train.count(1))
    print("3 star reviews in training set: ", y_train.count(2))
    print("4 star reviews in training set: ", y_train.count(3))
    print("5 star reviews in training set: ", y_train.count(4))

    # print how many reviews with 1 star are in the validation set
    print("1 star reviews in validation set: ", y_val.count(0))
    print("2 star reviews in validation set: ", y_val.count(1))
    print("3 star reviews in validation set: ", y_val.count(2))
    print("4 star reviews in validation set: ", y_val.count(3))
    print("5 star reviews in validation set: ", y_val.count(4))

    # print how many reviews with 1 star are in the test set
    print("1 star reviews in test set: ", y_test.count(0))
    print("2 star reviews in test set: ", y_test.count(1))
    print("3 star reviews in test set: ", y_test.count(2))
    print("4 star reviews in test set: ", y_test.count(3))
    print("5 star reviews in test set: ", y_test.count(4))



    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    model_builder = BertModelBuilder(num_labels=5)
    logger.info("Starting model training")
    model_builder.train(X_train, y_train, X_val, y_val)
    logger.info("Model training completed")
    model_builder.predict(X_test, y_test)