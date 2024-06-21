from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import torch

class LLaMAModel:
    def __init__(self, model_name, num_labels):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def _load_dataset(self, file_path):
        data_files = {"test": file_path}
        return load_dataset("csv", data_files=data_files)

    def _preprocess_dataset(self, dataset):
        def preprocess_function(examples):
            return self.tokenizer(examples['review'], truncation=True)

        return dataset.map(preprocess_function, batched=True)

    def zero_shot_predict(self, file_path):
        dataset = self._load_dataset(file_path)
        tokenized_dataset = self._preprocess_dataset(dataset)

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir="./results"),
            tokenizer=self.tokenizer,
        )

        predictions = trainer.predict(tokenized_dataset["test"]).predictions
        predicted_labels = torch.argmax(torch.tensor(predictions), dim=-1)
        return predicted_labels

    def predict(self, reviews):
        inputs = self.tokenizer(reviews, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions

# Esempio di utilizzo della classe per zero-shot learning
if __name__ == "__main__":
    model_name = "nome-del-modello-llama3"
    num_labels = 2  # Cambia secondo le tue etichette

    llama_model = LLaMAModel(model_name, num_labels)

    # Zero-shot prediction su un dataset di test reale
    test_file_path = "path/to/your/test_data_real.csv"
    zero_shot_predictions = llama_model.zero_shot_predict(test_file_path)
    print(zero_shot_predictions)

    # Predizioni su nuove recensioni
    new_reviews = ["Questo prodotto è fantastico!", "Non mi piace per niente."]
    predictions = llama_model.predict(new_reviews)
    print(predictions)
