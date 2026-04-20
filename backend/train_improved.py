"""
Improved Training Script for Text Classification Model
Fixes data leakage, adds proper validation, and prevents overfitting.
"""

import os
import pandas as pd                                                
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import warnings
warnings.filterwarnings('ignore')

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model", "classifier_v2")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "data.csv")
BATCH_SIZE = 4
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LENGTH = 256
EARLY_STOPPING_PATIENCE = 2

LABEL_TO_ID = {"human": 0, "ai": 1, "humanized": 2}
ID_TO_LABEL = {0: "human", 1: "ai", 2: "humanized"}
NUM_LABELS = 3


def load_and_preprocess_data():
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Original dataset: {len(df)} samples")
    
    # Remove duplicates to fix data leakage
    duplicates = df[df.duplicated(subset='text', keep=False)]
    print(f"Found {len(duplicates)} duplicate texts")
    df = df.drop_duplicates(subset='text', keep='first')
    print(f"After removing duplicates: {len(df)} samples")
    
    # Add hard examples
    hard_examples = [
        {"text": "traffic is high", "label": 0},
        {"text": "traffic congestion is high", "label": 1},
        {"text": "traffic is kinda high today", "label": 2},
        {"text": "yeah traffic is bad lol", "label": 0},
        {"text": "due to increased vehicles, congestion is high", "label": 1},
        {"text": "traffic's pretty rough today, isn't it?", "label": 2},
        {"text": "i love cooking", "label": 0},
        {"text": "cooking is a beneficial activity", "label": 1},
        {"text": "i really enjoy cooking when i have time", "label": 2},
        {"text": "my cat is cute", "label": 0},
        {"text": "feline companionship provides emotional benefits", "label": 1},
        {"text": "my cat is such a sweet little buddy", "label": 2},
    ]
    hard_df = pd.DataFrame(hard_examples)
    df = pd.concat([df, hard_df], ignore_index=True)
    print(f"After adding hard examples: {len(df)} samples")
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 0]
    df['label'] = df['label'].astype(int)
    
    # Create stratified splits: 80% train, 10% val, 10% test
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Check for overlap between splits
    train_texts = set(train_df['text'])
    test_texts = set(test_df['text'])
    overlap = train_texts.intersection(test_texts)
    if len(overlap) > 0:
        print(f"WARNING: {len(overlap)} overlapping texts between train and test!")
    else:
        print("No overlap between train and test sets")
    
    return (
        Dataset.from_pandas(train_df[['text', 'label']]),
        Dataset.from_pandas(val_df[['text', 'label']]),
        Dataset.from_pandas(test_df[['text', 'label']])
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": accuracy, "f1": f1}


def train_model():
    print("=" * 50)
    print("Improved Text Classification Model Training")
    print("=" * 50)
    
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data()
    
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )
    tokenized_test = test_dataset.map(
        lambda examples: tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )
    
    print(f"\nLoading model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=50,
        no_cuda=True,
        fp16=False,
        dataloader_num_workers=0,
        seed=42,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Detailed evaluation on test set
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    
    predictions = trainer.predict(tokenized_test)
    true_labels = predictions.label_ids
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)
    print(f"Labels: {target_names}")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    train_model()