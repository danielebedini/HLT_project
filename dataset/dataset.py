import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_concatenate_csv_files(directory):
    dfs = []
    for rating in range(1, 6):
        file_path = os.path.join(directory, f'reviews_{rating}_stars.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
    if not dfs:
        raise ValueError("No files found in the directory to concatenate.")
    return pd.concat(dfs, ignore_index=True)

def balance_dataset(df, max_samples_per_category):
    min_samples_per_category = min(df['overall'].value_counts().min(), max_samples_per_category)
    balanced_dfs = []
    for rating in range(1, 6):
        df_rating = df[df['overall'] == rating]
        balanced_dfs.append(df_rating.sample(min(len(df_rating), min_samples_per_category), random_state=1))
    return pd.concat(balanced_dfs, ignore_index=True)

def split_and_save_dataset(df, output_dir, max_samples_per_category, test_size=0.2, val_size=0.1, class_ratios=None):
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['overall'], random_state=1)
    
    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['overall'], random_state=1)
    
    # Save validation set
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    
    # Create balanced test set
    balanced_test_df = balance_dataset(test_df, max_samples_per_category)
    balanced_test_df.to_csv(os.path.join(output_dir, 'test_balanced.csv'), index=False)
    
    # Create imbalanced test set
    if class_ratios:
        imbalanced_test_dfs = []
        total_samples = len(test_df)
        for rating in range(1, 6):
            df_rating = test_df[test_df['overall'] == rating]
            ratio = class_ratios.get(rating, 0)
            num_samples = int(total_samples * ratio)
            # Adjust num_samples if there are not enough samples in the class
            num_samples = min(len(df_rating), num_samples)
            imbalanced_test_dfs.append(df_rating.sample(num_samples, random_state=1))
        imbalanced_test_df = pd.concat(imbalanced_test_dfs, ignore_index=True)
        imbalanced_test_df.to_csv(os.path.join(output_dir, 'test_imbalanced.csv'), index=False)
    else:
        print("No class ratios provided for imbalanced test set.")

    # Save training set
    train_df.to_csv(os.path.join(output_dir, 'training.csv'), index=False)
    
    # Print statistics
    print("Statistiche del dataset di training:")
    print(train_df['overall'].value_counts())
    print("\nStatistiche del dataset di validation:")
    print(val_df['overall'].value_counts())
    print("\nStatistiche del dataset di test bilanciato:")
    print(balanced_test_df['overall'].value_counts())
    if class_ratios:
        print("\nStatistiche del dataset di test sbilanciato:")
        print(imbalanced_test_df['overall'].value_counts())

def create_combined_balanced_dataset(dataset1_dir, dataset2_dir, output_dir, max_samples_per_category, class_ratios):
    # Carica i dataset
    df1 = load_and_concatenate_csv_files(dataset1_dir)
    df2 = load_and_concatenate_csv_files(dataset2_dir)
    
    # Combina i dataset
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Crea dataset bilanciato
    balanced_combined_df = balance_dataset(combined_df, max_samples_per_category)
    
    # Split e salva i dataset
    split_and_save_dataset(balanced_combined_df, output_dir, max_samples_per_category, class_ratios=class_ratios)

# Percorsi dei dataset
dataset1_dir = 'dataset/dataset_1/'
dataset2_dir = 'dataset/dataset_2/'
output_dir = 'dataset/'
max_samples_per_category = 1500


class_ratios = {
    1: 0.1,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.6
}

if __name__ == '__main__':
    create_combined_balanced_dataset(dataset1_dir, dataset2_dir, output_dir, max_samples_per_category, class_ratios)
