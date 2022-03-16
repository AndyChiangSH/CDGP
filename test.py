import json

with open("./datasets/DGen/total_new_cleaned_test.json", "r") as file:
    questions = json.load(file)

for question in questions:
    sent = question["sentence"].replace("**blank**", "[MASK]")
    answer = question["answer"]
    print(sent)
    print(answer)
    print(sent + " [SEP] " + answer)