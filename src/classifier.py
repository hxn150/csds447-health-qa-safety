import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset

LABEL2ID = {"SAFE": 0, "UNSAFE": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class SafetyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data():
    data_dir = Path('data/safety_classifier')
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    return train_df, val_df, test_df

def evaluate_model(model, test_dataset):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            item = test_dataset[i]
            inputs = {
                'input_ids': item['input_ids'].unsqueeze(0).to(model.device),
                'attention_mask': item['attention_mask'].unsqueeze(0).to(model.device)
            }
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
            true_labels.append(item['labels'].item())
    
    return predictions, true_labels

def main():
    train_df, val_df, test_df = load_data()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(train_df['label'].value_counts().to_dict())
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    train_labels = [LABEL2ID[l] for l in train_df['label']]
    val_labels = [LABEL2ID[l] for l in val_df['label']]
    test_labels = [LABEL2ID[l] for l in test_df['label']]
    
    train_dataset = SafetyDataset(train_df['text'].tolist(), train_labels, tokenizer)
    val_dataset = SafetyDataset(val_df['text'].tolist(), val_labels, tokenizer)
    test_dataset = SafetyDataset(test_df['text'].tolist(), test_labels, tokenizer)
    
    output_dir = Path('models/safety_classifier')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\nTraining...")
    trainer.train()
    
    print("\nEvaluating...")
    predictions, true_labels = evaluate_model(model, test_dataset)
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['SAFE', 'UNSAFE']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, predictions)
    print("        SAFE  UNSAFE")
    for i, label in enumerate(['SAFE', 'UNSAFE']):
        print(f"{label:6}  {cm[i][0]:4}  {cm[i][1]:6}")
    
    model.save_pretrained(output_dir / 'final')
    tokenizer.save_pretrained(output_dir / 'final')
    
    print(f"\nSaved to: {output_dir / 'final'}")

if __name__ == "__main__":
    main()