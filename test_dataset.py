import json

# with open("./datasets/DGen/total_new_cleaned_test.json", "r") as file:
#     datas = json.load(file)

# print("test:", len(datas))

# with open("./datasets/DGen/total_new_cleaned_train.json", "r") as file:
#     datas = json.load(file)

# print("train:", len(datas))

with open("./results/result_RoBERTa_DGen_model1.json", "r") as file:
    datas = json.load(file)

print("result:", len(datas))

"""
train: 2321
test: 259
result: 
"""