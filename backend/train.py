"""
Training Script for Text Classification Model
Fine-tunes distilgpt2 on a custom dataset for classifying text as:
- human (human-written)
- ai (AI-generated)
- humanized (AI text edited by humans)

This script is optimized for CPU training with small batch sizes.
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
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

# Configuration
MODEL_NAME = "distilgpt2"  # Using distilgpt2 for better CPU performance
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "model", "classifier")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset", "data.csv")

# Training hyperparameters optimized for CPU
BATCH_SIZE =2  # Small batch size for CPU
NUM_EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_LENGTH = 256

# Label mappings
LABEL_TO_ID = {"human": 0, "ai": 1, "humanized": 2}
ID_TO_LABEL = {0: "human", 1: "ai", 2: "humanized"}
NUM_LABELS = 3


def load_and_preprocess_data():
    """
    Load the dataset and preprocess it for training.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Loading dataset...")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Please create the dataset first."
        )
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    print("Raw dataset preview:")
    print(df.head())

    print("Columns:", df.columns)
    print("Shape:", df.shape)
    
    # Clean data
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.len() > 0]
    
    # Map labels to integers
    df['label'] = df['label'].astype(int)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split into train and validation
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    return train_dataset, val_dataset


def tokenize_function(examples, tokenizer):
    """
    Tokenize the text data.
    
    Args:
        examples: Batch of examples from the dataset
        tokenizer: The tokenizer to use
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Evaluation prediction object
        
    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }


def train_model():
    """
    Main training function.
    """
    print("=" * 50)
    print("Text Classification Model Training")
    print("=" * 50)
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: CPU (no_cuda=True)")
    print("=" * 50)
    
    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data()
    
    # Load tokenizer
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token (distilgpt2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    
    # Set pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - optimized for CPU
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
        logging_steps=10,
        no_cuda=True,  # Force CPU usage
        fp16=False,  # Disable mixed precision for CPU
        dataloader_num_workers=0,  # Single process for CPU
        seed=42,
        save_total_limit=2,  # Keep only 2 best models
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save final model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Print classification report
    print("\nGenerating classification report...")
    predictions = trainer.predict(tokenized_val)
    true_labels = predictions.label_ids
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    
    print("\nClassification Report:")
    target_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]
        # Print classification report
    print("\nGenerating classification report...")

    predictions = trainer.predict(tokenized_val)
    true_labels = predictions.label_ids
    pred_labels = np.argmax(predictions.predictions, axis=-1)

    print("\nClassification Report:")

    # Debug
    print("True labels:", set(true_labels))
    print("Pred labels:", set(pred_labels))

    labels = list(ID_TO_LABEL.keys())
    target_names = [ID_TO_LABEL[i] for i in labels]

    print(classification_report(
        true_labels,
        pred_labels,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    train_model()