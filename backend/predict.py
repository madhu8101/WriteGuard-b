"""
Prediction Module
Handles text classification using the trained model.
"""

import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict, Tuple
import numpy as np

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier")

# Label mappings
ID_TO_LABEL = {0: "human", 1: "ai", 2: "humanized"}
LABEL_TO_ID = {"human": 0, "ai": 1, "humanized": 2}

# Global variables to cache model and tokenizer
_model = None
_tokenizer = None


class TextClassifier:
    """
    Text classifier for distinguishing between human-written, AI-generated,
    and humanized AI text.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the trained model directory. 
                       Defaults to backend/model/classifier
        """
        if model_path is None:
            model_path = MODEL_PATH
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")  # Force CPU usage
        
    def load_model(self):
        """
        Load the trained model and tokenizer from disk.
        """
        if self.model is not None and self.tokenizer is not None:
            return  # Already loaded
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train.py"
            )
        
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model - force CPU usage
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32  # Use float32 for CPU compatibility
        )
        
        # Move to CPU explicitly
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def predict(self, text: str) -> Dict[str, any]:
        """
        Predict the class of a given text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict containing 'label' and 'confidence'
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)
        
        # Convert to Python types
        label = ID_TO_LABEL[predicted_class.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities for detailed output
        all_probabilities = probabilities.cpu().numpy()
        class_probabilities = {
            ID_TO_LABEL[i]: float(prob) 
            for i, prob in enumerate(all_probabilities)
        }
        
        return {
            "label": label,
            "confidence": round(confidence_score, 4),
            "probabilities": class_probabilities
        }
    
    def predict_batch(self, texts: list) -> list:
        """
        Predict classes for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]


# Singleton instance for global use
_classifier = None


def get_classifier() -> TextClassifier:
    """
    Get the global classifier instance, creating it if needed.
    
    Returns:
        TextClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = TextClassifier()
        _classifier.load_model()
    return _classifier


def predict_text(text: str) -> Dict[str, any]:
    """
    Convenience function to predict the class of a text.
    
    Args:
        text: Input text to classify
        
    Returns:
        Dict containing 'label' and 'confidence'
    """
    classifier = get_classifier()
    return classifier.predict(text)