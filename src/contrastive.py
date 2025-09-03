import nlpaug.augmenter.word as naw
import pandas as pd
import numpy as np

aug = naw.SynonymAug(aug_src='wordnet')

def generate_contrastive_pairs(df, num_pairs=1000):
    """Generate contrastive pairs (positive: augmented same-label, negative: different-label)."""
    positive_pairs = []
    negative_pairs = []
    racist = df[df['label'] == 1]['clean_text'].values
    non_racist = df[df['label'] == 0]['clean_text'].values
    
    for _ in range(num_pairs // 2):
        if len(racist) > 0:
            text = np.random.choice(racist)
            aug_text = aug.augment(text)[0]  
            positive_pairs.append((text, aug_text, 1))
        if len(non_racist) > 0:
            text = np.random.choice(non_racist)
            aug_text = aug.augment(text)[0]
            positive_pairs.append((text, aug_text, 1))
    
    for _ in range(num_pairs // 2):
        if len(racist) > 0 and len(non_racist) > 0:
            text1 = np.random.choice(racist)
            text2 = np.random.choice(non_racist)
            negative_pairs.append((text1, text2, 0))
    
    contrastive_df = pd.DataFrame(positive_pairs + negative_pairs, columns=['text1', 'text2', 'similarity'])
    return contrastive_df