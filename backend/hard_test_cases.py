"""
Adversarial Testing Script
Tests the model on hard cases to verify robustness.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from predict import TextClassifier

def run_hard_tests():
    """Run adversarial tests on the model."""
    
    # Hard test cases - similar sentences across different classes
    test_cases = [
        # Traffic examples
        {"text": "traffic is high", "expected": "human"},
        {"text": "traffic congestion is high", "expected": "ai"},
        {"text": "traffic is kinda high today", "expected": "humanized"},
        {"text": "yeah traffic is bad lol", "expected": "human"},
        {"text": "due to increased vehicles, congestion is high", "expected": "ai"},
        {"text": "traffic's pretty rough today, isn't it?", "expected": "humanized"},
        
        # Cooking examples
        {"text": "i love cooking", "expected": "human"},
        {"text": "cooking is a beneficial activity", "expected": "ai"},
        {"text": "i really enjoy cooking when i have time", "expected": "humanized"},
        
        # Cat examples
        {"text": "my cat is cute", "expected": "human"},
        {"text": "feline companionship provides emotional benefits", "expected": "ai"},
        {"text": "my cat is such a sweet little buddy", "expected": "humanized"},
        
        # Additional hard cases
        {"text": "idk man", "expected": "human"},
        {"text": "I do not know, man", "expected": "humanized"},
        {"text": "uncertainty prevails", "expected": "ai"},
        
        {"text": "lol that was funny", "expected": "human"},
        {"text": "that was quite humorous", "expected": "humanized"},
        {"text": "the content elicited amusement", "expected": "ai"},
    ]
    
    print("=" * 60)
    print("ADVERSARIAL TESTING - Hard Cases")
    print("=" * 60)
    
    # Try v2 model first, fall back to v1
    model_path = os.path.join(os.path.dirname(__file__), "model", "classifier_v2")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "model", "classifier")
    
    classifier = TextClassifier(model_path=model_path)
    classifier.load_model()
    
    correct = 0
    total = len(test_cases)
    
    for case in test_cases:
        text = case["text"]
        expected = case["expected"]
        
        result = classifier.predict(text)
        predicted = result["label"]
        confidence = result["confidence"]
        
        is_correct = predicted == expected
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"
        
        print(f"\n{status}")
        print(f"  Text: {text}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted} (confidence: {confidence:.2%})")
        print(f"  All probs: {result['probabilities']}")
    
    print("\n" + "=" * 60)
    print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print("=" * 60)
    
    return correct, total


if __name__ == "__main__":
    run_hard_tests()