import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
class LLaMAModel:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def predict(self, X):
        # Tokenize the input
        inputs = self.tokenizer(X, padding=True, truncation=True, return_tensors="pt")
        # Convert input IDs and attention mask to Long type
        inputs = {key: value.to(torch.long) for key, value in inputs.items()}
        # Perform the prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Get the predicted class
        predicted_class = torch.argmax(outputs.logits, dim=1)
        return predicted_class

if __name__ == "__main__":
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HUGGINGFACE_API_TOKEN"] = "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"

    # Load your pre-trained model
    num_labels = 5  # For ratings from 1 to 5
    model = LLaMAModel("RLHFlow/ArmoRM-Llama3-8B-v0.1", num_labels)

    # take data from the datasets (save in data_2.py)
    from data_2 import X_train, X_test, y_train, y_test, X_test_balanced, y_test_balanced, X_test_real, y_test_real 

    test_texts = X_test_real
    
    # Optional: Add a prompt (e.g., "Review: " before each text)
    prompt = "Review: "
    test_texts = [prompt + text for text in test_texts]
    
    # Make predictions
    print("Predicting...")
    predictions = model.predict(test_texts)
    print(predictions)
    
    # calculate the classification_report
    from sklearn.metrics import classification_report
    import pandas as pd

    y_pred = predictions
    y_test = y_test_real

    # print classification report
    print(classification_report(y_test, y_pred))
