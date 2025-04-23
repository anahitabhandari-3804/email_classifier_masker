import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import joblib

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EmailClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self, X, y, model_path, epochs=1, batch_size=16):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        os.makedirs(model_path, exist_ok=True)
        joblib.dump(label_encoder, os.path.join(model_path, 'label_encoder.pkl'))

        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        train_dataset = EmailDataset(X_train, y_train, self.tokenizer)
        val_dataset = EmailDataset(X_val, y_val, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=os.path.join(model_path, 'checkpoints'),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=os.path.join(model_path, 'logs'),
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)

    def predict(self, texts):
        label_encoder = joblib.load(os.path.join(self.model_path, 'label_encoder.pkl'))

        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

        return label_encoder.inverse_transform(predictions.cpu().numpy())

    def load_model(self, model_path):
        self.model_path = model_path
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model.to(self.device)

def train_model(data_path: str, model_path: str) -> None:
    df = pd.read_csv(data_path).sample(n=2000, random_state=42)  # limit dataset to 2000 rows for faster training
    X = df['masked_email'].tolist()
    y = df['type'].tolist()
    num_classes = len(set(y))
    classifier = EmailClassifier(num_labels=num_classes)
    classifier.train(X, y, model_path)