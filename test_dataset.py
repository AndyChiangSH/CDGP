import json

with open(r"./datasets/total_new_cleaned_test.json", "r") as file:
    datas = json.load(file)

print("test:", len(datas))

with open(r"./datasets/total_new_cleaned_train.json", "r") as file:
    datas = json.load(file)

print("train:", len(datas))