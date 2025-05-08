from pathlib import Path
from transformers import AutoTokenizer,BertTokenizer ,BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertModel:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        
        "Nếu train lân đầu: dùng 2 dòng dưới và cmt 3 dòng tiếp theo, nếu muốn dùng pretrained model sau khi train lần đầu, cmt 2 dòng dưới và bỏ cmt 3 dòng tiếp theo"
       
        model_path = Path("./results/checkpoint-4287").resolve()
        if model_path.exists():
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # Hoặc model phù hợp nếu bạn đang dùng loại khác
            self.model = BertForSequenceClassification.from_pretrained(str(model_path), num_labels=num_labels, local_files_only=True)
        
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        

        self.model.to(device)
        self.trainer = None

    def preprocess(self, texts, labels):
        df = {"text": texts, "label": labels}
        dataset = Dataset.from_dict(df)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        dataset = dataset.map(tokenize_function, batched=True)
        dataset = dataset.rename_column("label", "labels")
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return dataset

    def train(self, train_texts, train_labels, num_epochs=3, batch_size=8):
        train_dataset = self.preprocess(train_texts, train_labels)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        self.trainer.train()

    def predict(self, texts, batch_size=8):
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            texts = texts.tolist()

        self.model.eval()
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            tokens = self.tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**tokens)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1)
                predictions.extend(batch_predictions.cpu().tolist())

        return predictions
  