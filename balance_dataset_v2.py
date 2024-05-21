import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, input_file, down_sample_size=300, random_state=42, test_size=0.2):
        self.input_file = input_file
        self.down_sample_size = down_sample_size
        self.random_state = random_state
        self.test_size = test_size
        self.data = None
        self.train_data_balanced = None
        self.test_data_unbalanced = None

    def load_data(self):
        """Load the data from the input file."""
        self.data = pd.read_csv(self.input_file)
        print("Data loaded successfully.")

    def split_data(self):
        """Split the dataset into balanced training set and unbalanced test set."""
        X = self.data.drop(columns=['overall'])  # Features
        y = self.data['overall']  # Target

        # Split data into training (balanced) and test (unbalanced)
        X_train_balanced, X_test_unbalanced, y_train_balanced, y_test_unbalanced = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Combine X_train_balanced and y_train_balanced back into a DataFrame
        self.train_data_balanced = pd.concat([X_train_balanced, y_train_balanced], axis=1)

        # Set the test_data_unbalanced as the unbalanced test set
        self.test_data_unbalanced = pd.concat([X_test_unbalanced, y_test_unbalanced], axis=1)

        print("Dataset split into balanced training set and unbalanced test set.")

    def down_sample_data(self, star_rating=5.0):
        """Down-sample the five-star reviews in the balanced training set."""
        five_star_reviews = self.train_data_balanced[self.train_data_balanced['overall'] == star_rating]
        not_five_star_reviews = self.train_data_balanced[self.train_data_balanced['overall'] != star_rating]

        # Perform down-sampling only on the five-star reviews in the balanced training set
        down_sampled = five_star_reviews.sample(n=self.down_sample_size, random_state=self.random_state)
        self.train_data_balanced = pd.concat([not_five_star_reviews, down_sampled])
        
        print("Down-sampling of five-star reviews in the balanced training set completed.")

    def save_data(self, train_output_file, test_output_file):
        """Save the balanced training set and unbalanced test set to output files."""
        self.train_data_balanced.to_csv(train_output_file, index=False)
        self.test_data_unbalanced.to_csv(test_output_file, index=False)
        print(f"Balanced training set saved successfully in {train_output_file}.")
        print(f"Unbalanced test set saved successfully in {test_output_file}.")

    def check_overlaps(self):
        """Check for overlaps between the balanced training set and unbalanced test set."""
        # Check for overlaps between the balanced training set and unbalanced test set
        overlap = self.train_data_balanced.merge(self.test_data_unbalanced, on=['reviewText', 'overall'], how='inner')
        print(f"Number of overlapping reviews: {len(overlap)}")

    def print_len_reviews(self,):
        # print 1 star reviews
        print("Number of 1 star reviews: ", len(self.test_data_unbalanced[self.test_data_unbalanced['overall'] == 1.0]))
        # print 2 star reviews
        print("Number of 2 star reviews: ", len(self.test_data_unbalanced[self.test_data_unbalanced['overall'] == 2.0]))
        # print 3 star reviews
        print("Number of 3 star reviews: ", len(self.test_data_unbalanced[self.test_data_unbalanced['overall'] == 3.0]))
        # print 4 star reviews
        print("Number of 4 star reviews: ", len(self.test_data_unbalanced[self.test_data_unbalanced['overall'] == 4.0]))
        # print 5 star reviews
        print("Number of 5 star reviews: ", len(self.test_data_unbalanced[self.test_data_unbalanced['overall'] == 5.0]))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def plot_cake_graph(dataset, title):
        """Plot cake graph of a given dataset."""
        # Plot cake graph
        dataset['overall'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, title=title)
        plt.axis('equal')
        plt.show()

    # Usage of the class
    processor = DataProcessor('amazon_reviews.csv', down_sample_size=300)
    processor.load_data()
    processor.split_data()
    processor.down_sample_data(star_rating=5.0)
    processor.down_sample_data(star_rating=4.0)
    processor.check_overlaps()
    processor.print_len_reviews()
    processor.save_data('balanced_train_data.csv', 'unbalanced_test_data.csv')
    # print("Overlaps: ", processor.check_overlaps())
    plot_cake_graph(processor.train_data_balanced, 'Balanced Training Set')
    plot_cake_graph(processor.test_data_unbalanced, 'Unbalanced Test Set')

    """
    print("Balanced training set:")
    print(processor.train_data_balanced.head())

    print("\nUnbalanced test set:")
    print(processor.test_data_unbalanced.head())
    """