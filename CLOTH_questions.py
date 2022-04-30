# 三個錯誤 + SEP + 正確選項
from click import option
import nltk
import json
import os


en2num = {"A": 0, "B": 1, "C": 2, "D": 3}
num2en = ["A", "B", "C", "D"]
    
with open("./datasets/CLOTH/test/high/high3624.json", "r") as f:
    dataset = json.load(f)

sent = dataset["article"]
answer_sent = sent
options = dataset["options"]
answers = dataset["answers"]

# print(sent)
# print(options)
# print(answers)

space_count = sent.count(" _ ")

for i in range(space_count):
    answer_sent = answer_sent.replace(" _ ", f"[{options[i][en2num[answers[i]]]}]", 1)
    sent = sent.replace(" _ ", f"[{i+1}]", 1)
 
print(answer_sent)   
print(sent)

for i, option in enumerate(options):
    print(f"{i+1}.")
    for j, op in enumerate(option):
        print(f"({num2en[j]}) {op}")
        
print(f"answer: ")
for i, answer in enumerate(answers):
    print(f"{i+1}. {answer}")