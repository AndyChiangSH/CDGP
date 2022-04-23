# 三個錯誤 + SEP + 正確選項
import nltk
from tqdm import tqdm
import json
import os


CLOTH_PATH = "./datasets/CLOTH"

en2num = {"A": 0, "B": 1, "C": 2, "D": 3}
question_list = list()

for tv in ["test"]:
    tv_path = os.path.join(CLOTH_PATH, tv)
    for grade in ["high", "middle"]:
        grade_path = os.path.join(tv_path, grade)
        print(grade_path)

        files = os.listdir(grade_path)
        # print(files)

        for file in tqdm(files):
            if not file.endswith(".json"):
                continue
            
            full_path = os.path.join(grade_path, file)
            # print(full_path)
            
            with open(full_path, "r") as f:
                dataset = json.load(f)

            sents = nltk.sent_tokenize(dataset["article"])
            options = dataset["options"]
            answers = dataset["answers"]
            # print(sents)
            # print(options)
            # print(answers)

            i = 0
            for sent in sents:
                if "_" in sent:
                    # print(sent)
                    blank_num = sent.count("_")
                    blank_texts = sent.split("_")
                    for j in range(blank_num):
                        sentence = ""
                        k = 0
                        for blank_text in blank_texts:
                            if k == blank_num:
                                answer_index = en2num[answers[i+j]]
                                dis_list = list()
                                for l in range(len(options[i+j])):
                                    if l == answer_index:
                                        answer = options[i+j][l]
                                    else:
                                        dis_list.append(options[i+j][l])                            
                                # sentence
                                sentence += blank_text
                                
                                question = {
                                    "answer": answer,
                                    "distractors": dis_list,
                                    "sentence": sentence
                                }
                                # print(question)
                                question_list.append(question)

                                break
                            
                            # 得到ans的索引值
                            ans_index = en2num[answers[i+k]]                         
                            if k == j:  # 要被mask的
                                sentence += blank_text + "[MASK]"
                            else: # 其他不是被mask的
                                sentence += blank_text + options[i+k][ans_index]               
                            k += 1
                            
                    i += blank_num


with open("CLOTH_cleaned_test.json", "w") as file:
    json.dump(question_list, file)

print(f"CLOTH question number = {len(question_list)}")    
print("Done!")