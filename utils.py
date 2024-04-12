import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "could've": "could have",
    # we can add more if needed
}

def expand_contractions(text):
    # Replace contractions using the contractions dictionary
    for contraction, expansion in contractions_dict.items():
        text = text.replace(contraction, expansion)
    return text

def preprocess_text(text):
    if isinstance(text, str):  # if the value is a string
        # tokenizer
        tokens = word_tokenize(text.lower())
        
        # remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # lemmatisation
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # reconstruct the text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    else:
        return ''  # if the value is not a string, return an empty string


def preprocess_text_porter_stemmer(text):
    if isinstance(text, str):
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization and/or Stemming
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # tokens = [stemmer.stem(token) for token in tokens]  # Uncomment for stemming
        
        # Filter out tokens that are too short or non-alphanumeric
        tokens = [token for token in tokens if len(token) > 1 and token.isalnum()]
        
        # Reconstruct the text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    else:
        return ''

# preprocess with removing unrelevant words

def preprocess_text_v2(text):
    if isinstance(text, str):
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter out tokens that are too short or non-alphanumeric
        tokens = [token for token in tokens if len(token) > 1 and token.isalnum()]
        
        # Reconstruct the text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    else:
        return ''

# preprocess with removing contractions
def preprocess_text_contractions(text):
    if isinstance(text, str):
        # Expand contractions
        text = expand_contractions(text)
        
        # Tokenization
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Filter out tokens that are too short or non-alphanumeric
        tokens = [token for token in tokens if len(token) > 1 and token.isalnum()]
        
        # Reconstruct the text
        preprocessed_text = ' '.join(tokens)
        
        return preprocessed_text
    else:
        return ''


# save the results to a json file
def save_results_json(model, accuracy, class_report, filename):
    """
    Save results to a JSON file.

    Args:
        model: The trained model or its configuration.
        accuracy (float): The accuracy score.
        class_report (str): The classification report.
        filename (str): Name of the JSON file to save.

    Returns:
        None
    """

    # Serialize model if it's not a string
    model_info = str(model) if not isinstance(model, str) else model
    
    # Prepare data for JSON serialization
    results = {
        'model': model_info,
        'accuracy': accuracy,
        'classification_report': class_report
    }
    
    # Save results to JSON file
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
