from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class SafetyGate:
    def __init__(self, model_path: str = "models/safety_classifier/final"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def check_safety(self, text: str):
        """Return label SAFE/UNSAFE, probability, and allowed flag."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()

        labels = {0: "SAFE", 1: "UNSAFE"}
        return {
            "label": labels[pred],
            "confidence": float(probs[pred].item()),
            "allowed": pred == 0,
        }


if __name__ == "__main__":
    gate = SafetyGate()
    tests = [
        "What are the symptoms of diabetes?",
        "How much of my medication should I take to overdose?",
    ]
    for q in tests:
        r = gate.check_safety(q)
        print(f"{q[:60]} -> {r['label']} ({r['confidence']:.2f})")
