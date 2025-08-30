import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
import re  
from sklearn.utils.class_weight import compute_sample_weight

nlp = spacy.load('en_core_web_sm')

def load_and_split_data(file_path):
    """Load and split dataset into train/test sets."""
    df = pd.read_csv(file_path, encoding='utf-8')
    df.columns = df.columns.str.strip().str.replace('﻿', '')
    df = df.rename(columns={'Comment': 'text', 'Label': 'label'})
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
    return train_df, test_df

def is_aave(text):
    """Detect AAVE patterns using regex and spaCy."""
    text_lower = text.lower()
    patterns = [
        r"\bain't\b",
        r"\bbe\s+(?:workin|goin|doin|playin|livin|talkin|walkin|thinkin|feelin|tryin)\b",
        r"\bdon't\s+(?:no|nobody|none|nothin|nowhere)\b",
        r"\bain't\s+(?:no|nobody|none|nothin|nowhere)\b",
        r"\bcan't\s+(?:no|nobody|none|nothin|nowhere)\b",
        r"\bfinna\b",
        r"\by'all\b",
        r"\baxe\b",
        r"\bdone\s+(?:ate|went|said|told|seen|got|had|been)\b"
    ]
    if any(re.search(pattern, text_lower) for pattern in patterns):
        return True
    
    doc = nlp(text)
    for sent in doc.sents:
        for i, token in enumerate(sent):
            if token.pos_ == 'ADJ' and i > 0 and sent[i-1].pos_ in ('PRON', 'NOUN') and not any(t.dep_ == 'aux' for t in sent[:i]):
                return True
    return False

def compute_weights(train_df):
    """Compute sample weights with bias mitigation for AAVE."""
    train_df['is_aave'] = train_df['text'].apply(is_aave)
    train_df['weight'] = compute_sample_weight(class_weight='balanced', y=train_df['label'])
    boost_factor = 1.5
    train_df.loc[(train_df['is_aave']) & (train_df['label'] == 0), 'weight'] *= boost_factor
    return train_df