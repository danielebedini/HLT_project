from utils import plot_accuracy, plot_dataset_data, plot_pie_graph
from data import DataPreprocessor

not_balanced_data = DataPreprocessor('amazon_reviews.csv')
not_balanced_data.load_and_preprocess()
not_balanced_data.split_data()

preprocessor = DataPreprocessor('new_balanced_data.csv')
# Carica e preprocessa i dati
preprocessor.load_and_preprocess()
# Divide i dati
preprocessor.split_data()

plot_dataset_data(preprocessor.data)
plot_dataset_data(not_balanced_data.data)
plot_pie_graph(preprocessor.data)
plot_pie_graph(not_balanced_data.data)