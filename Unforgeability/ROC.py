import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import numpy as np
from scipy import interpolate
import torch
from scipy.interpolate import make_interp_spline
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    GenerationConfig,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import roc_curve
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}


tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
model = RobertaForSequenceClassification.from_pretrained('./roberta-base', num_labels=2).to(device)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


batch_size = 16

with open('../watermark_text/Ours/6B_beam.json', 'r') as file:
    data = json.load(file)

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts,
    labels,
    test_size=0.3
)


train_dataset = TextDataset(train_texts, train_labels)
val_dataset = TextDataset(val_texts, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


num_epochs = 3
optimizer = optim.Adam(model.parameters(), lr=2e-5)


model.train()
for epoch in range(num_epochs):
    for batch_texts, batch_labels in train_dataloader:
        optimizer.zero_grad()
        encodings = tokenizer(
            batch_texts,
            padding=True,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        labels_tensor = torch.tensor(batch_labels).to(device)
        outputs = model(**encodings, labels=labels_tensor)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')


scores = []
model.eval()
with torch.no_grad():
    for batch_text, batch_labels in val_dataloader:
        encodings = tokenizer(
            batch_text,
            padding=True,
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        outputs = model(**encodings)
        batch_scores = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        scores.extend(batch_scores)


fpr, tpr, thresholds = roc_curve(val_labels, scores)



def moving_average(data, window_size=2):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

fpr_new = np.linspace(fpr.min(), fpr.max(), 300)
tpr_smooth = interpolate.interp1d(fpr, tpr, kind='linear')(fpr_new)
fpr_new_d = moving_average(fpr_new, window_size=2)
tpr_smooth_d = moving_average(tpr_smooth, window_size=2)


fpr_new_d = np.insert(fpr_new_d, 0, 0)
tpr_smooth_d = np.insert(tpr_smooth_d, 0, 0)


if fpr_new_d[-1] < 1:
    fpr_new_d = np.append(fpr_new_d, 1)
    tpr_smooth_d = np.append(tpr_smooth_d, 1)


df = pd.DataFrame({'FPR': fpr_new_d, 'TPR': tpr_smooth_d})
df.to_csv('roc_data.csv', index=False)

auc_value = roc_auc_score(val_labels, scores)
print(f'AUC: {auc_value}')
