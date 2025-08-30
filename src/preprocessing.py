from symspellpy import SymSpell
import re
import contractions
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary('frequency_dictionary_en_82_765.txt', term_index=0, count_index=1)

def preprocess_text(text):
    """Clean and normalize text with SymSpell correction."""
    text = re.sub(r'<.*?>', '', text)
    text = contractions.fix(text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    return suggestions[0].term if suggestions else text

def get_bert_embeddings(texts, batch_size=32):
    """Generate BERT [CLS] embeddings for a list of texts."""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)
    return np.vstack(embeddings)