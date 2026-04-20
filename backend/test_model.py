"""
Test script to verify the trained model's ability to distinguish
between human, AI, and humanized text.
"""

from predict import TextClassifier

def test_model():
    print("Initializing Text Classifier...")
    classifier = TextClassifier()
    
    # Test cases for each category
    test_cases = {
        "Human Text (with errors/informal)": [
            "i really enjoy baking on weekends, its relaxing",
            "my cat is the cutest thing ever, she always makes me smile :)",
            "i dunno, i just feel like sometimes things work out",
            "we went to the beach last weekend and it was so fun!! the water was perfect",
            "i love cooking but im not very good at it tbh",
            "the traffic was terrible today, took me forever to get home",
            "i hate when my phone battery dies so quickly",
            "my roommate ate my leftovers and didnt even say anything smh",
        ],
        "AI-Generated Text (formal/corporate)": [
            "Moreover, the synergy between process and technology is instrumental in driving strategic value delivery across cross-sectoral verticals.",
            "In summary, organizations must leverage holistic platform to navigate the complexities of risk mitigation effectively.",
            "It is essential to note that performance benchmarking plays a crucial role in optimizing comprehensive platform across multiple cross-sectorals.",
            "A comprehensive analysis of process automation reveals pivotal insights into architecture optimization and integrated operational performance.",
            "The utilization of blueprint in enterprise contexts provides a robust foundation for agile systemic resilience generation.",
            "Furthermore, it is crucial to accelerate a comprehensive framework that transforms the blueprint in a structured manner.",
            "Leveraging mechanism allows for the facilitation of scalable outcomes, thereby enhancing ROI at an organizational level.",
            "The robust implementation of system frameworks ensures robust scalability and streamlined utilization of resources.",
        ],
        "Humanized AI Text (polished but personal)": [
            "In my experience, taking small steps with drawing tends to energizes things over time.",
            "It seems like learning kind of comes off as pretty eye-opening at first, but it actually creates real change.",
            "Honestly, there's something about setting small goals that just appears meaningful in a quiet way.",
            "In my experience, morning walks is one of those things that gets kind of beautiful once you start.",
            "Surprisingly, I've found that spending time outside doesn't require as much planning as I thought.",
            "In my experience, focus is actually more nuanced than expected once you approach it with curiosity.",
            "Honestly, it took me a while to understand that resilience is somewhat surprisingly fun by nature.",
            "I've realized that patience can actually makes more sense if you give it enough time and space.",
        ]
    }
    
    print("\n" + "="*70)
    print("Testing Text Classification Model")
    print("="*70)
    
    for category, texts in test_cases.items():
        print(f"\n{category}:")
        print("-" * 50)
        
        # Determine expected label based on category
        if "Humanized AI" in category:
            expected_label = "humanized"
        elif "AI-Generated" in category:
            expected_label = "ai"
        elif "Human Text" in category:
            expected_label = "human"
        else:
            expected_label = "unknown"
        
        for text in texts:
            result = classifier.predict(text)
            predicted_label = result['label']
            confidence = result['confidence']
            
            status = "✓ CORRECT" if predicted_label == expected_label else "✗ WRONG"
            print(f"{status} Text: {text[:60]}...")
            print(f"  Predicted: {predicted_label} ({confidence:.2%})")
            print()
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

if __name__ == "__main__":
    test_model()