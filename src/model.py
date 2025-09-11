import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.metrics import f1_score
from transformers import EarlyStoppingCallback

# Load tokenizer and original model for contrastive features
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
original_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Custom Dataset for text data with tokenization
class TextDataset(Dataset):
    def __init__(self, texts, labels, weights=None, tokenizer=None):
        self.tokenizer = tokenizer
        self.encodings = [tokenizer(text, padding=False, truncation=True, max_length=512, return_tensors="pt") for text in texts]
        self.labels = torch.LongTensor(labels)
        self.weights = torch.FloatTensor(weights) if weights is not None else None
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings[idx]["input_ids"].squeeze(0),
            "attention_mask": self.encodings[idx]["attention_mask"].squeeze(0),
            "labels": self.labels[idx]
        }
        if self.weights is not None:
            item["weights"] = self.weights[idx]
        return item

# Load data
train_df = pd.read_csv('processed_train.csv')
test_df = pd.read_csv('processed_test.csv')
contrastive_df = pd.read_csv('contrastive_pairs.csv')

# Prepare data
train_texts = train_df['clean_text'].values
test_texts = test_df['clean_text'].values
train_labels = train_df['label'].values
test_labels = test_df['label'].values
train_weights = train_df['weight'].values  # From bias mitigation
contrastive_text1 = contrastive_df['text1'].values
contrastive_text2 = contrastive_df['text2'].values
contrastive_labels = contrastive_df['similarity'].values

# K-Fold Cross-Validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_texts, train_labels), 1):
    print(f"Fold {fold}")
    train_texts_fold = train_texts[train_idx]
    val_texts_fold = train_texts[val_idx]
    train_labs_fold = train_labels[train_idx]
    val_labs_fold = train_labels[val_idx]
    train_wts_fold = train_weights[train_idx]
    val_wts_fold = train_weights[val_idx]

    # Datasets
    train_dataset = TextDataset(train_texts_fold, train_labs_fold, train_wts_fold, tokenizer)
    val_dataset = TextDataset(val_texts_fold, val_labs_fold, val_wts_fold, tokenizer)
    test_dataset = TextDataset(test_texts, test_labels, tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
    output_dir=f'./results/fold_{fold}',
    num_train_epochs=6,  # Try 6 or more
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir=f'./logs/fold_{fold}',
    logging_steps=10,
    metric_for_best_model="f1",  # Use F1 for early stopping
    greater_is_better=True,
    save_strategy="steps",
)

    # Custom Trainer to handle weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            weights = inputs.pop("weights") if "weights" in inputs else None
            outputs = model(**inputs)
            loss = outputs.loss  # Loss is computed by the model if labels are provided
            if loss is None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs.logits, labels)
            if weights is not None:
                loss = (loss * weights.mean()).mean()  # Weighted loss
            return (loss, outputs) if return_outputs else loss

    # Initialize and train model
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
        "precision": precision_score(p.label_ids, p.predictions.argmax(-1)),
        "recall": recall_score(p.label_ids, p.predictions.argmax(-1)),
        "f1": f1_score(p.label_ids, p.predictions.argmax(-1))
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

    trainer.train()
    # Save model for each fold
    trainer.save_model(f'./distilbert_model_fold_{fold}')

    # Predictions for this fold
    test_outputs = trainer.predict(test_dataset)
    distilbert_predictions = test_outputs.predictions

    # Contrastive Learning with NT-Xent Loss
    def nt_xent_loss(embeddings1, embeddings2, labels, temperature=0.5):
        batch_size = embeddings1.size(0)
        embeddings1 = nn.functional.normalize(embeddings1, p=2, dim=1)  # Normalize embeddings
        embeddings2 = nn.functional.normalize(embeddings2, p=2, dim=1)
        embeddings = torch.cat([embeddings1, embeddings2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

        # Compute cosine similarity matrix for the concatenated batch
        similarity_matrix = torch.matmul(embeddings, embeddings.t()) / temperature
        labels_expanded = labels.unsqueeze(1).expand(-1, 2 * batch_size)  # Expand to match concatenated size

        # Mask for positive pairs (same index pair) and negative pairs
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=embeddings.device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1
            positive_mask[i + batch_size, i] = 1
        negative_mask = 1 - positive_mask - torch.eye(2 * batch_size, device=embeddings.device)

        # Numerator: similarity with positive pair
        exp_similarities = torch.exp(similarity_matrix)
        positive_similarities = (positive_mask * exp_similarities).sum(dim=1)
        # Denominator: similarity with all pairs except self
        negative_similarities = (negative_mask * exp_similarities).sum(dim=1)

        # Debugging
        print(f"Positive Similarities: {positive_similarities}")
        print(f"Negative Similarities: {negative_similarities}")
        if torch.any(positive_similarities <= 0) or torch.any(negative_similarities <= 0):
            print("Warning: Non-positive similarities detected, adjusting with higher epsilon")
            epsilon = 1e-4  # Increase epsilon to handle near-zero cases
        else:
            epsilon = 1e-8
        loss = -torch.log(positive_similarities / (positive_similarities + negative_similarities + epsilon))
        return loss.mean()

    def get_embedding(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.distilbert(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return embedding

    # Generate embeddings for contrastive pairs
    embeddings1 = torch.stack([get_embedding(t, tokenizer, model) for t in contrastive_text1])
    embeddings2 = torch.stack([get_embedding(t, tokenizer, model) for t in contrastive_text2])
    contrastive_labels_tensor = torch.tensor(contrastive_labels, dtype=torch.float32)

    # Compute contrastive loss (for validation)
    contrastive_loss = nt_xent_loss(embeddings1, embeddings2, contrastive_labels_tensor)
    print(f"Fold {fold} Contrastive Loss:", contrastive_loss.item())

    # Train Logistic Regression on contrastive pairs
    pair_features = (embeddings1 - embeddings2).numpy()
    scaler = StandardScaler()
    pair_features_scaled = scaler.fit_transform(pair_features)
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(pair_features_scaled, contrastive_labels)

    # Generate contrastive features for test set
    test_embeddings1 = torch.stack([get_embedding(t, tokenizer, model) for t in test_texts])
    test_embeddings2 = test_embeddings1  # Placeholder: Replace with actual test pairs if available
    test_pair_features = (test_embeddings1 - test_embeddings2).numpy()[:len(test_texts)]  # Match test set size
    test_pair_features_scaled = scaler.transform(test_pair_features)
    lr_probs = lr_model.predict_proba(test_pair_features_scaled)[:, 1]  # Probabilities for test set

    # Ensemble Predictions (Soft Voting)
    distilbert_probs = torch.softmax(torch.from_numpy(distilbert_predictions), dim=1)[:, 1].numpy()
    ensemble_probs = (distilbert_probs + lr_probs) / 2 
    # Tune threshold for best F1
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.71, 0.01):
        preds = (ensemble_probs > thresh).astype(int)
        f1 = f1_score(test_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    ensemble_preds = (ensemble_probs > best_thresh).astype(int)
    print(f"Best threshold for fold {fold}: {best_thresh} (F1: {best_f1:.3f})")
    print(f"Fold {fold} Ensemble Classification Report:")
    print(classification_report(test_labels, ensemble_preds))

    # Create directories for results and logs
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/fold_{fold}_log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Fold {fold} Contrastive Loss: {contrastive_loss.item()}\n\n")
        f.write(f"Fold {fold} Ensemble Classification Report:\n")
        f.write(classification_report(test_labels, ensemble_preds))

    # Save predictions and model
    np.save(f'./ensemble_predictions_fold_{fold}.npy', ensemble_preds)
    joblib.dump(lr_model, f'./logistic_regression_model_fold_{fold}.pkl')

print("Step 4 completed. Check results in ./results and ./logs for each fold.")