import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import json
import pandas as pd
import matplotlib.pyplot as plt

from contractions import contractions_dict

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#custom_stopwords = ['phone', 'card', 'SanDisk', 'memory', 'stick', 'drive', 'usb', 'device', 'product', 'item', 'thing']

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
        #stop_words.update(custom_stopwords)
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

def plot_dataset_data(data):
    """
    Plot the distribution of the dataset.

    Args:
        data: The dataset to plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    data['overall'].value_counts().sort_index().plot(kind='bar')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

def plot_pie_graph(data):
    """
    Plot the distribution of the dataset.

    Args:
        data: The dataset to plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    data['overall'].value_counts().sort_index().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    # plot confusion matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix:", cm)
    cm_df = pd.DataFrame(cm, index=[i for i in range(1, 6)], columns=[i for i in range(1, 6)])
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def get_wordcloud(data, overall):
    from wordcloud import WordCloud

    # Give the cloud only 1 star reviews
    # change the value to 2, 3, 4, or 5 to see the word cloud for other ratings
    data = data[data['overall'] == overall]['reviewText'].values

    # Create a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data))
    
    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def oversampler(X_train, y_train):
    from imblearn.over_sampling import RandomOverSampler

    oversampler = RandomOverSampler(random_state=42)
    X_train, y_train = oversampler.fit_resample(X_train.values.reshape(-1, 1), y_train)
    X_train = X_train.flatten()

    return X_train, y_train
