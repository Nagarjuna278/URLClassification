import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

class URLDataset(Dataset):
    def __init__(self, urls, labels, tokenizer, max_length):
        self.urls = urls
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        url = self.urls[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            url,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'url_text': url,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RoBERTaCNN(nn.Module):
    def __init__(self, roberta_model, num_classes, max_length):
        super(RoBERTaCNN, self).__init__()
        self.roberta = roberta_model
        self.conv1 = nn.Conv1d(768, 256, kernel_size=3)
        self.conv2 = nn.Conv1d(256, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)[0]
        roberta_output = roberta_output.permute(0, 2, 1)
        conv1_output = torch.relu(self.conv1(roberta_output))
        conv2_output = torch.relu(self.conv2(conv1_output))
        pooled = self.pool(conv2_output).squeeze(2)
        output = self.fc(self.dropout(pooled))
        return output

class RoBERTaCNNURLClassifier:
    def __init__(self, model_name='roberta-base', num_classes=4, max_length=128, learning_rate=2e-5):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        roberta_model = RobertaModel.from_pretrained(model_name)
        self.model = RoBERTaCNN(roberta_model, num_classes, max_length)
        self.max_length = max_length
        self.learning_rate = learning_rate

    @staticmethod
    def prepare_data(csv_path):
        df_sample = pd.read_csv(csv_path)
        label_mapping = {'benign': 0, 'defacement': 1, 'malware': 2, 'phishing': 3}
        df_sample['label'] = df_sample['type'].map(label_mapping)
        df_sample['combined_url'] = df_sample.apply(lambda row: f"{row['url']} | IP: {row['ip_address']} | nameserver: {row['nameservers']} | KL Divergence: {row['kl_divergence']} | Entropy: {row['entropy']} | Digit-Letter Ratio: {row['digit_letter_ratio']} | TLD Count: {row['top_level_domains_count']} | Dash Count: {row['dash_count']} | URL Length: {row['url_length']} | Digits in Domain: {row['digits_in_domain']} | Suspicious Words Count: {row['suspicious_words_count']} | Subdomains Count: {row['subdomains_count']} | Brand Name Modified: {row['brand_name_modified']} | Long Hostname Phishy: {row['long_hostname_phishy']} | Punctuation Symbols Count: {row['punctuation_symbols_count']} | Colons in Hostname Count: {row['colons_in_hostname_count']} | IP Address or Hexadecimal: {row['ip_address_or_hexadecimal']} | Vowel-Consonant Ratio: {row['vowel_consonant_ratio']} | Short Hostname Phishy: {row['short_hostname_phishy']} | At Symbol: {row['at_symbol']}", axis=1)
        
        min_count = df_sample['label'].value_counts().min()
        df_balanced = df_sample.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=32)).reset_index(drop=True)
        
        urls = df_sample['combined_url'].values
        labels = df_sample['label'].values
        return train_test_split(urls, labels, test_size=0.1, random_state=42)

    def train(self, train_dataloader, test_dataloader, epochs=8, rank=0):
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model = DDP(self.model, device_ids=[rank])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            total_loss = 0
            count = 0
            for batch in train_dataloader:
                count += 1
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                if count % 1000 == 0:
                    print(f"Epoch {epoch + 1}, Batch {count}")
            print(f"Loss: {total_loss / len(train_dataloader)}")
            if rank == 0:
                self.evaluate(test_dataloader, device)

    def evaluate(self, dataloader, device):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['benign', 'defacement', 'malware', 'phishing']))

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pth'))
        self.tokenizer.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    classifier = RoBERTaCNNURLClassifier()
    train_urls, test_urls, train_labels, test_labels = classifier.prepare_data('extracted_features.csv')
    full_dataset = URLDataset(np.concatenate((train_urls, test_urls)), np.concatenate((train_labels, test_labels)), classifier.tokenizer, classifier.max_length)
    train_dataset = Subset(full_dataset, range(len(train_urls)))
    test_dataset = Subset(full_dataset, range(len(train_urls), len(full_dataset)))
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    classifier.train(train_dataloader, test_dataloader, epochs=4, rank=rank)
    
    if rank == 0:
        print("\nEvaluating on test set:")
        classifier.evaluate(test_dataloader, torch.device('cuda:0'))
        save_dir = 'roberta_cnn_url_classifier'
        classifier.save_model(save_dir)
    
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)