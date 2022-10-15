import nltk
nltk.download('punkt')
from tqdm.notebook import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
import torch
import matplotlib.pyplot as plt
import json
import re
import os
# load data
input_list = list()
label_list = list()

with open("./CLOTH/train_valid_input.txt", "r") as f:
  for line in f:
    input_list.append(line[:-1])

with open("./CLOTH/train_valid_label.txt", "r") as f:
  for line in f:
    label_list.append(line[:-1])

# 建立dataset dataloader
BATCH_SIZE = 16
EPOCH = 1
data_dic = {"input": input_list, "label": label_list}
dataset = Dataset.from_dict(data_dic)
print(len(dataset))

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# fine-tune bart

# from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import Trainer, TrainingArguments
# from transformers import BertTokenizer, BertForMaskedLM
# import torch

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)
# start finetune
# 進度條
num_training_steps = EPOCH * len(train_loader)
progress_bar = tqdm(range(num_training_steps))

# 開始訓練
loss_history = []
for epoch in range(EPOCH):
  for batch in train_loader:
    inputs = tokenizer.batch_encode_plus(batch["input"], truncation=True, padding="max_length", max_length=50, return_tensors="pt")
    labels = tokenizer.batch_encode_plus(batch["label"], truncation=True, padding="max_length", max_length=50, return_tensors="pt")["input_ids"]
    # print(inputs)
    # print(labels)
    output = model(**inputs.to(device), labels=labels.to(device))
    optimizer.zero_grad()
    loss = output.loss
    logits = output.logits
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    progress_bar.update(1)
  
  print(f"[epoch {epoch+1}] loss: {loss.item()}")

# show result
print(loss_history)
print(len(loss_history))
# paint training loss graph
# import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.title('Training loss')
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['loss'], loc='upper right')
plt.show()

# savemodel
dir_path = r"./cloze_bart_model"

if not os.path.exists(dir_path):
  os.mkdir(dir_path)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(os.path.join(dir_path, "bart_mask_model1"))