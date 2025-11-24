from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SafetyGate:
    def __init__(self, model_path='models/safety_classifier/final'):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
    
    def check_safety(self, question):
        inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()
        
        labels = {0: "SAFE", 1: "UNSAFE", 2: "REFUSAL"}
        
        return {
            "label": labels[pred],
            "confidence": probs[pred].item(),
            "allowed": pred == 0 
        }

if __name__ == "__main__":
    gate = SafetyGate()
    
    test_cases = [
        "What are symptoms of diabetes?",
        "How do I overdose on aspirin?",
        "Can you diagnose my chest pain?"
    ]
    
    for q in test_cases:
        result = gate.check_safety(q)
        print(f"{q[:40]}... → {result['label']} ({result['confidence']:.2f})")