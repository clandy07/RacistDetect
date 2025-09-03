import numpy as np
import os
from src.data_utils import load_and_split_data, compute_weights
from src.preprocessing import preprocess_text, get_bert_embeddings
from src.contrastive import generate_contrastive_pairs

# Paths
data_path = os.path.join('data', 'RacismDetectionDataSet.csv')

# Load and split data
train_df, test_df = load_and_split_data(data_path)

# Compute weights with bias mitigation
train_df = compute_weights(train_df)

# Preprocess text
train_df['clean_text'] = train_df['text'].apply(preprocess_text)
test_df['clean_text'] = test_df['text'].apply(preprocess_text)
train_df.to_csv('processed_train.csv', index=False)
test_df.to_csv('processed_test.csv', index=False)

# Generate BERT embeddings
train_embeddings = get_bert_embeddings(train_df['clean_text'].tolist())
test_embeddings = get_bert_embeddings(test_df['clean_text'].tolist())
np.save('train_embeddings.npy', train_embeddings)
np.save('test_embeddings.npy', test_embeddings)


print("Data preparation completed. Check saved files.")