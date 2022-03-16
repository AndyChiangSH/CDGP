import json

with open("./datasets/DGen/total_new_cleaned_test.json", "r") as file:
    datas = json.load(file)

print("test:", len(datas))

with open("./datasets/DGen/total_new_cleaned_train.json", "r") as file:
    datas = json.load(file)

print("train:", len(datas))

"""
train: 2321
test: 259
"""