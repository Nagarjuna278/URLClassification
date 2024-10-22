import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, Subset
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
            return_token_type_ids=False,
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

class BERTURLClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=4, max_length=128, learning_rate=2e-5):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.max_length = max_length
        self.learning_rate = learning_rate

    @staticmethod
    def prepare_data(csv_path):
        df = pd.read_csv(csv_path)
        df_sample = df.sample(n=100000, random_state=32)
        label_mapping = {'benign': 0, 'defacement': 1, 'malware': 2, 'phishing': 3}
        df_sample['label'] = df_sample['type'].map(label_mapping)
        
        # Combine URL, IP, and DNS into a single feature
        df_sample['combined_url'] = df_sample.apply(lambda row: f"{row['url']} | IP: {row['ip_address']} | nameserver: {row['nameservers']} |  KL Divergence: {row['kl_divergence']} | Entropy: {row['entropy']} | Digit-Letter Ratio: {row['digit_letter_ratio']} | TLD Count: {row['top_level_domains_count']} | Dash Count: {row['dash_count']} | URL Length: {row['url_length']} | Digits in Domain: {row['digits_in_domain']} | Suspicious Words Count: {row['suspicious_words_count']} | Subdomains Count: {row['subdomains_count']} | Brand Name Modified: {row['brand_name_modified']} | Long Hostname Phishy: {row['long_hostname_phishy']} | Punctuation Symbols Count: {row['punctuation_symbols_count']} | Colons in Hostname Count: {row['colons_in_hostname_count']} | IP Address or Hexadecimal: {row['ip_address_or_hexadecimal']} | Vowel-Consonant Ratio: {row['vowel_consonant_ratio']} | Short Hostname Phishy: {row['short_hostname_phishy']} | At Symbol: {row['at_symbol']}",axis=1)
        #df_sample['combined_url'] = df_sample.apply(lambda row: f"{row['url']} | IP: {row['ip_address']} | nameserver: {row['nameservers']}",axis=1)

        # Get the minimum number of samples for each type
        min_count = df_sample['label'].value_counts().min()

        # Sample an equal number from each class
        df_balanced = df_sample.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=32)).reset_index(drop=True)
        
        # Use combined_url instead of just url
        urls = df_sample['combined_url'].values
        labels = df_sample['label'].values

        return train_test_split(urls, labels, test_size=0.2, random_state=42)

    def train(self, train_dataloader, test_dataloader, epochs=8):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        self.model.train()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            count=0
            for batch in train_dataloader:
                count+=1
                print(count," ",len(train_dataloader)," ",epoch+1)
                optimizer.zero_grad()
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Loss: {total_loss / len(train_dataloader)}")

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = self.model(input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=['benign', 'defacement', 'malware', 'phishing']))


    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Save the model
        self.model.save_pretrained(save_dir)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        print(f"Model saved to {save_dir}")


def main():
    classifier = BERTURLClassifier()

    # Load data and prepare datasets
    train_urls, test_urls, train_labels, test_labels = classifier.prepare_data('extracted_features.csv')
    
    full_dataset = URLDataset(np.concatenate((train_urls, test_urls)), 
                              np.concatenate((train_labels, test_labels)), 
                              classifier.tokenizer, 
                              classifier.max_length)

    # Create train and test datasets
    train_dataset = Subset(full_dataset, range(len(train_urls)))
    test_dataset = Subset(full_dataset, range(len(train_urls), len(full_dataset)))

    # DataLoader for batching
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    # Train the model
    classifier.train(train_dataloader, test_dataloader, epochs=3)

    # Evaluate the model
    print("\nEvaluating on test set:")
    classifier.evaluate(test_dataloader)

    # Save the model
    save_dir = 'bert_url_classifier'
    classifier.save_model(save_dir)

if __name__ == "__main__":
    main()