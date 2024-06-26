from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

class LLaMAModel:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
    def predict(self, X):
        # Tokenize the input
        inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")
        # Perform the prediction
        outputs = self.model(**inputs)
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1)
        return predicted_class

    def zero_shot_learning(self, X, labels):
        # Tokenize the input
        inputs = self.tokenizer(X, labels, padding=True, truncation=True, return_tensors="pt")
        # Perform the prediction
        outputs = self.model(**inputs)
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1)
        return predicted_class


# Esempio di utilizzo della classe per zero-shot learning
if __name__ == "__main__":

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HUGGINGFACE_API_TOKEN"] = "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"

    # load data from data_2
    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced, X_test_real, y_test_real
    from utils import preprocess_text_v2

    # print type of X_test_real[:0]
    # print(type(X_test_real[:0]))

    # Convert the reviews to a string
    X_test_balanced = [str(review) for review in X_test_balanced]
    X_test_real = [str(review) for review in X_test_real]

    # preprocess data
    X_test_balanced = preprocess_text_v2(X_test_balanced)
    X_test_real = preprocess_text_v2(X_test_real)

    # Llama model
    num_labels = 5
    model = LLaMAModel("RLHFlow/ArmoRM-Llama3-8B-v0.1", num_labels)
    
    # Test on balanced data
    predictions = model.predict(X_test_balanced)
    print(predictions)

    # Test on real data
    predictions = model.predict(X_test_real)
    print(predictions)