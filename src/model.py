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
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV

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
    distilbert_pred_labels = distilbert_predictions.argmax(-1)
    
    # ADD: Get validation predictions separately
    val_outputs = trainer.predict(val_dataset)
    distilbert_val_predictions = val_outputs.predictions
    
    distilbert_report = classification_report(test_labels, distilbert_pred_labels)

    # Step 1: Pre-train DistilBERT on contrastive pairs (similarity task)
    print(f"Fold {fold} Starting contrastive pre-training...")

    # Extend contrastive dataset
    contrastive_texts = list(contrastive_text1) + list(contrastive_text2)
    contrastive_labels_extended = list(contrastive_labels) + list(contrastive_labels)
    contrastive_dataset = TextDataset(contrastive_texts, contrastive_labels_extended, tokenizer=tokenizer)

    # Initialize contrastive model
    contrastive_model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    )
    contrastive_model.config.output_hidden_states = True

    # Custom contrastive trainer with similarity-based loss
    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)

            # Use CLS token
            embeddings = outputs.hidden_states[-1][:, 0, :]  # [CLS]
            batch_size = embeddings.size(0) // 2
            embeddings1 = nn.functional.normalize(embeddings[:batch_size], p=2, dim=1)
            embeddings2 = nn.functional.normalize(embeddings[batch_size:], p=2, dim=1)

            similarity = torch.sum(embeddings1 * embeddings2, dim=1)
            contrastive_loss = -torch.mean(similarity * labels[:batch_size])

            return (contrastive_loss, outputs) if return_outputs else contrastive_loss

    # Training arguments (slightly longer epochs + optional batch size increase)
    contrastive_training_args = TrainingArguments(
        output_dir=f'./contrastive_results/fold_{fold}',
        num_train_epochs=5,
        per_device_train_batch_size=16,   # increased batch size
        per_device_eval_batch_size=16,
        logging_dir=f'./contrastive_logs/fold_{fold}',
        logging_steps=50,
        save_strategy="no",
    )

    contrastive_trainer = ContrastiveTrainer(
        model=contrastive_model,
        args=contrastive_training_args,
        train_dataset=contrastive_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    contrastive_trainer.train()

    # ==========================
    # Step 1b: Fine-Tune on Classification Labels
    # ==========================
    train_dataset_cls = TextDataset(train_texts_fold, train_labs_fold, tokenizer=tokenizer)
    val_dataset_cls = TextDataset(val_texts_fold, val_labs_fold, tokenizer=tokenizer)

    training_args_cls = TrainingArguments(
        output_dir=f'./finetune_results/fold_{fold}',
        num_train_epochs=5,                 # slightly longer fine-tuning
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir=f'./finetune_logs/fold_{fold}',
        logging_steps=50,
        save_strategy="no",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer_cls = Trainer(
        model=contrastive_model,
        args=training_args_cls,
        train_dataset=train_dataset_cls,
        eval_dataset=val_dataset_cls,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=lambda p: {"f1": f1_score(p.label_ids, p.predictions.argmax(-1))}
    )
    trainer_cls.train()

    print(f"Fold {fold} Contrastive pre-training & fine-tuning completed.")

    # ==========================
    # Step 2: Generate Embeddings
    # ==========================
    def get_embedding(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.distilbert(**inputs)
            # Use mean pooling over last 4 layers for richer embeddings
            last_4 = outputs.hidden_states[-4:]
            cls_tokens = torch.stack([layer[:,0,:] for layer in last_4])  # shape [4, batch, hidden]
            embedding = torch.mean(cls_tokens, dim=0).squeeze()            # mean over layers
        return embedding

    train_embeddings = torch.stack([get_embedding(t, tokenizer, contrastive_model) for t in train_texts_fold])
    train_features = train_embeddings.numpy()
    test_embeddings = torch.stack([get_embedding(t, tokenizer, contrastive_model) for t in test_texts])
    test_features = test_embeddings.numpy()

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    # ==========================
    # Step 3: Logistic Regression
    # ==========================
    from sklearn.linear_model import LogisticRegressionCV
    lr_model = LogisticRegressionCV(
        cv=5,
        max_iter=5000,
        Cs=np.logspace(-4, 4, 10),        # finer regularization grid
        solver='saga',
        scoring='f1',
        class_weight='balanced',
        penalty='elasticnet',
        l1_ratios=[0.2, 0.4, 0.6, 0.8],        # try different mixes
        n_jobs=-1,
    )
    lr_model.fit(train_features_scaled, train_labs_fold)
    
    # Calibrate the logistic regression model for better probability estimates
    print(f"Fold {fold} Calibrating Logistic Regression probabilities...")
    lr_model_calibrated = CalibratedClassifierCV(lr_model, cv=3, method='sigmoid')  # Changed: removed .best_estimator_
    lr_model_calibrated.fit(train_features_scaled, train_labs_fold)

    # ==========================
    # Step 4: Evaluate
    # ==========================
    # Enhanced Ensemble Predictions (Weighted Soft Voting)
    # 1. Compute validation embeddings & scale
    val_embeddings = torch.stack([get_embedding(t, tokenizer, contrastive_model) for t in val_texts_fold])
    val_features = val_embeddings.numpy()
    val_features_scaled = scaler.transform(val_features)

    # 2. Logistic Regression probabilities on validation (use calibrated model)
    lr_val_probs = lr_model_calibrated.predict_proba(val_features_scaled)[:, 1]

    # 3. Find best threshold for LR on validation
    precision, recall, thresholds = precision_recall_curve(val_labs_fold, lr_val_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_lr_threshold = thresholds[np.argmax(f1_scores)]

    # 4. LR validation predictions & F1
    lr_val_preds = (lr_val_probs > best_lr_threshold).astype(int)
    lr_val_f1 = f1_score(val_labs_fold, lr_val_preds)

    # 5. DistilBERT probabilities on validation
    distilbert_val_probs = torch.softmax(
        torch.from_numpy(distilbert_val_predictions), dim=1
    )[:, 1].numpy()
    
    # Apply temperature scaling for DistilBERT probability calibration
    print(f"Fold {fold} Applying temperature scaling to DistilBERT probabilities...")
    def find_best_temperature(probs, labels):
        """Find optimal temperature for probability calibration using negative log-likelihood"""
        from scipy.optimize import minimize_scalar
        def nll_loss(T):
            scaled_probs = np.clip(probs ** (1/T), 1e-8, 1-1e-8)
            return -np.mean(labels * np.log(scaled_probs) + (1-labels) * np.log(1-scaled_probs))
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        return result.x
    
    temperature = find_best_temperature(distilbert_val_probs, val_labs_fold)
    distilbert_val_probs_calibrated = distilbert_val_probs ** (1/temperature)
    
    # Find best threshold for DistilBERT on validation
    precision_db, recall_db, thresholds_db = precision_recall_curve(val_labs_fold, distilbert_val_probs_calibrated)
    f1_scores_db = 2 * precision_db * recall_db / (precision_db + recall_db + 1e-8)
    best_db_threshold = thresholds_db[np.argmax(f1_scores_db)]
    
    distilbert_val_preds = (distilbert_val_probs_calibrated > best_db_threshold).astype(int)
    distilbert_val_f1 = f1_score(val_labs_fold, distilbert_val_preds)

    # 6. Compute ensemble weights using multiple methods (all use soft voting)
    print(f"Fold {fold} Computing soft voting ensemble weights...")
    epsilon = 1e-6
    
    # Method 1: F1-based weighting (proportional to F1 performance)
    total_f1_val = distilbert_val_f1 + lr_val_f1 + epsilon
    distilbert_weight_f1 = distilbert_val_f1 / total_f1_val
    lr_weight_f1 = lr_val_f1 / total_f1_val
    
    # Method 2: Grid search for optimal soft voting weights
    print(f"Fold {fold} Grid searching for optimal soft voting weights...")
    best_grid_f1 = 0
    best_weights = (0.5, 0.5)
    
    for w_db in np.arange(0.0, 1.01, 0.05):  # 21 weight combinations
        w_lr = 1.0 - w_db
        # Soft voting: weighted combination of probabilities
        ensemble_val_probs_grid = w_db * distilbert_val_probs_calibrated + w_lr * lr_val_probs
        
        # Find best threshold for this weight combination
        prec, rec, thresh = precision_recall_curve(val_labs_fold, ensemble_val_probs_grid)
        f1_grid = 2 * prec * rec / (prec + rec + 1e-8)
        max_f1_grid = np.max(f1_grid)
        
        if max_f1_grid > best_grid_f1:
            best_grid_f1 = max_f1_grid
            best_weights = (w_db, w_lr)
    
    distilbert_weight_grid, lr_weight_grid = best_weights
    
    # Cross-validate the weight methods to choose the best one
    print(f"Fold {fold} Cross-validating soft voting weight methods...")
    def evaluate_soft_voting_weights(w_db, w_lr):
        """Evaluate soft voting ensemble with given weights"""
        ensemble_probs = w_db * distilbert_val_probs_calibrated + w_lr * lr_val_probs
        prec, rec, thresh = precision_recall_curve(val_labs_fold, ensemble_probs)
        f1_scores_cv = 2 * prec * rec / (prec + rec + 1e-8)
        return np.max(f1_scores_cv)
    
    f1_method1 = evaluate_soft_voting_weights(distilbert_weight_f1, lr_weight_f1)
    f1_method2 = best_grid_f1
    
    # Select best method
    methods = {
        'F1-based Soft Voting': (distilbert_weight_f1, lr_weight_f1, f1_method1),
        'Grid Search Soft Voting': (distilbert_weight_grid, lr_weight_grid, f1_method2)
    }
    
    best_method = max(methods.items(), key=lambda x: x[1][2])
    method_name, (distilbert_weight, lr_weight, method_f1) = best_method
    
    print(f"Fold {fold} Selected Method: {method_name}")
    print(f"Fold {fold} Method Comparison - F1-based: {f1_method1:.3f}, Grid Search: {f1_method2:.3f}")
    print(f"Fold {fold} Soft Voting Ensemble Weights - DistilBERT: {distilbert_weight:.3f}, LR: {lr_weight:.3f}")

    # 7. Compute test probabilities for both models (calibrated)
    distilbert_test_probs = torch.softmax(torch.from_numpy(distilbert_predictions), dim=1)[:, 1].numpy()
    distilbert_test_probs_calibrated = distilbert_test_probs ** (1/temperature)
    lr_test_probs = lr_model_calibrated.predict_proba(test_features_scaled)[:, 1]
    
    # Apply weights to test set probabilities (SOFT VOTING)
    ensemble_probs = distilbert_weight * distilbert_test_probs_calibrated + lr_weight * lr_test_probs

    # 8. Search for best ensemble threshold using validation set
    ensemble_val_probs = distilbert_weight * distilbert_val_probs_calibrated + lr_weight * lr_val_probs
    precision_ens, recall_ens, thresholds_ens = precision_recall_curve(val_labs_fold, ensemble_val_probs)
    f1_scores_ens = 2 * precision_ens * recall_ens / (precision_ens + recall_ens + 1e-8)
    best_ensemble_threshold = thresholds_ens[np.argmax(f1_scores_ens)]
    best_val_f1 = np.max(f1_scores_ens)

    # 9. Compute ensemble predictions on test set
    ensemble_preds = (ensemble_probs > best_ensemble_threshold).astype(int)
    ensemble_f1 = f1_score(test_labels, ensemble_preds)

    # 10. Apply best thresholds to test predictions for individual models
    distilbert_test_preds = (distilbert_test_probs_calibrated > best_db_threshold).astype(int)
    lr_test_preds = (lr_test_probs > best_lr_threshold).astype(int)
    
    # Recalculate reports with optimized thresholds
    distilbert_report = classification_report(test_labels, distilbert_test_preds)
    lr_report = classification_report(test_labels, lr_test_preds)
    ensemble_report = classification_report(test_labels, ensemble_preds)

    print(f"Best ensemble threshold for fold {fold}: {best_ensemble_threshold:.3f} (Val F1: {best_val_f1:.3f}, Test F1: {ensemble_f1:.3f})")
    print(f"Fold {fold} Temperature Scaling: {temperature:.3f}")
    print(f"Fold {fold} Soft Voting Model Weights - DistilBERT: {distilbert_weight:.3f}, LR: {lr_weight:.3f}")
    print(f"Fold {fold} Validation F1 Scores - DistilBERT: {distilbert_val_f1:.3f}, LR: {lr_val_f1:.3f}")
    print(f"Fold {fold} Test F1 Scores - DistilBERT: {f1_score(test_labels, distilbert_test_preds):.3f}, LR: {f1_score(test_labels, lr_test_preds):.3f}, Ensemble: {ensemble_f1:.3f}")
    print(f"Fold {fold} DistilBERT Classification Report:")
    print(distilbert_report)
    print(f"Fold {fold} Logistic Regression Classification Report:")
    print(lr_report)
    print(f"Fold {fold} Ensemble Classification Report:")
    print(ensemble_report)

    # Create directories for results and logs
    os.makedirs('logs', exist_ok=True)
    os.makedirs('contrastive_results', exist_ok=True)
    os.makedirs('contrastive_logs', exist_ok=True)
    
    log_path = f'logs/fold_{fold}_log.txt'
    with open(log_path, 'w') as f:
        f.write(f"Fold {fold} Results\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Ensemble Method: Soft Voting (Weighted Probability Combination)\n")
        f.write(f"Selected Weight Method: {method_name}\n")
        f.write(f"Method Comparison - F1-based: {f1_method1:.3f}, Grid Search: {f1_method2:.3f}\n")
        f.write(f"Temperature Scaling: {temperature:.3f}\n")
        f.write(f"Soft Voting Ensemble Weights - DistilBERT: {distilbert_weight:.3f}, LR: {lr_weight:.3f}\n")
        f.write(f"Best Thresholds - DistilBERT: {best_db_threshold:.3f}, LR: {best_lr_threshold:.3f}, Ensemble: {best_ensemble_threshold:.3f}\n\n")
        f.write(f"Validation F1 Scores:\n")
        f.write(f"  DistilBERT: {distilbert_val_f1:.3f}\n")
        f.write(f"  LR: {lr_val_f1:.3f}\n")
        f.write(f"  Ensemble (Soft Voting): {best_val_f1:.3f}\n\n")
        f.write(f"Test F1 Scores:\n")
        f.write(f"  DistilBERT: {f1_score(test_labels, distilbert_test_preds):.3f}\n")
        f.write(f"  LR: {f1_score(test_labels, lr_test_preds):.3f}\n")
        f.write(f"  Ensemble (Soft Voting): {ensemble_f1:.3f}\n\n")
        f.write(f"DistilBERT Classification Report:\n")
        f.write(distilbert_report + "\n\n")
        f.write(f"Logistic Regression Classification Report:\n")
        f.write(lr_report + "\n\n")
        f.write(f"Ensemble (Soft Voting) Classification Report:\n")
        f.write(ensemble_report)

    # Save predictions and models
    np.save(f'./ensemble_predictions_fold_{fold}.npy', ensemble_preds)
    joblib.dump(lr_model_calibrated, f'./logistic_regression_calibrated_model_fold_{fold}.pkl')
    
    # Save ensemble configuration
    ensemble_config = {
        'method': method_name,
        'voting_type': 'soft',
        'distilbert_weight': distilbert_weight,
        'lr_weight': lr_weight,
        'temperature': temperature,
        'best_threshold': best_ensemble_threshold,
        'distilbert_threshold': best_db_threshold,
        'lr_threshold': best_lr_threshold,
        'calibration': {
            'lr': 'CalibratedClassifierCV (sigmoid)',
            'distilbert': f'Temperature Scaling (T={temperature:.3f})'
        }
    }
    joblib.dump(ensemble_config, f'./ensemble_config_fold_{fold}.pkl')

print("Step 4 completed. Check results in ./results and ./logs for each fold.")