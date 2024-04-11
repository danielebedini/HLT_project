import pandas as pd

# Carica il dataset
data = pd.read_csv('amazon_reviews.csv')

# Filtra i pattern con 5 stelle
five_star_reviews = data[data['overall'] == 5.0]

# Esegui il down sampling a 300 pattern
down_sampled_five_star_reviews = five_star_reviews.sample(n=300, random_state=42)  # 'random_state' per riproducibilità

# Seleziona i pattern che non hanno 5 stelle
not_five_star_reviews = data[data['overall'] != 5.0]

# Unisci i down-sampled 5 stelle con gli altri pattern
final_data = pd.concat([not_five_star_reviews, down_sampled_five_star_reviews])

# Salva o procedi con l'analisi
final_data.to_csv('new_balanced_data.csv', index=False)