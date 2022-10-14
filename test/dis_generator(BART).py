"""
Distractor Generator (BART)
Author: AndyChiangSH
Time: 2022/10/14
"""

from tqdm import tqdm
import os
from transformers import BartTokenizer, BartForConditionalGeneration, pipeline
import numpy as np
import fasttext
import nltk
from nltk.tokenize import word_tokenize
import json


# Global variables
CSG_MODEL = "AndyChiang/cdgp-csg-bart-cloth"
DS_MODEL = "../models/DS/cdgp-ds-fasttext.bin"
DATASET = "../datasets/CLOTH/CLOTH_test_cleaned.json"
RESULT = "BART_CLOTH_model"
TOP_K = 10
STOP_WORDS = ["\n", ">", "<", ""]
WEIGHT = {"s0": 0.6, "s1": 0.15, "s2": 0.15, "s3": 0.1}
# WEIGHT = {"s0": 0.25, "s1": 0.25, "s2": 0.25, "s3": 0.25}
QUESTION_LIMIT = 100


def main():
    # Load CSG model
    print(f"Load CSG model at {CSG_MODEL}...")
    tokenizer = BartTokenizer.from_pretrained(CSG_MODEL)
    csg_model = BartForConditionalGeneration.from_pretrained(CSG_MODEL)

    # Create a unmasker
    unmasker = pipeline('fill-mask', tokenizer=tokenizer, model=csg_model, top_k=TOP_K)

    # Load DS model
    print(f"Load DS model at {DS_MODEL}...")
    ds_model = fasttext.load_model(DS_MODEL)

    # Load test dataset
    print(f"Test dataset at {DATASET}...")
    with open(DATASET, "r") as file:
        questions = json.load(file)

    # Generate distractors
    print("Generate distractors...")
    dis_results = list()
    i = 0
    for question in tqdm(questions):
        sent = question["sentence"].replace("[MASK]", "<mask>").replace("\n", "")
        answer = question["answer"]
        result = generate_dis(unmasker, ds_model, sent, answer)
        dis_result = {"distractors": question["distractors"], "generations": result}
        dis_results.append(dis_result)
        
        i += 1
        if i == QUESTION_LIMIT:
            break

    # Save result
    print(f"Save to 'result_{RESULT}.json'..")
    with open(f"./results/result_{RESULT}.json", "w") as file:
        json.dump(dis_results, file)

    print("Done!")


# Generate distractors of one question
def generate_dis(unmasker, ds_model, sent, answer):
    # Answer relating
    target_sent = sent + " <sep> " + answer

    # Generate Candidate Set
    cs = list()
    for cand in unmasker(target_sent):
        word = cand["token_str"].replace(" ", "").replace("\n", "")
        if len(word) > 0:  # Skip empty
            cs.append({"word": word, "s0": cand["score"], "s1": 0.0, "s2": 0.0, "s3": 0.0})
    
    # Confidence Score s0
    s0s = [c["s0"] for c in cs]
    new_s0s = min_max_y(s0s)

    for i, c in enumerate(cs):
        c["s0"] = new_s0s[i]
    
    # Word Embedding Similarity s1
    answer_vector = ds_model.get_word_vector(answer)
    word_similarities = list()
    for c in cs:
        c_vector = ds_model.get_word_vector(c["word"])
        word_similarity = similarity(answer_vector, c_vector)
        word_similarities.append(word_similarity)

    new_similarities = min_max_y(word_similarities)

    for i, c in enumerate(cs):
        c["s1"] = 1-new_similarities[i]

    # Contextual-Sentence Embedding Similarity s2
    correct_sent = sent.replace('<mask>', answer)
    correct_sent_vector = ds_model.get_sentence_vector(correct_sent)

    cand_sents = list()
    for c in cs:
        cand_sents.append(sent.replace('<mask>', c["word"]))

    sent_similarities = list()
    for cand_sent in cand_sents:
        cand_sent_vector = ds_model.get_sentence_vector(cand_sent)
        sent_similarity = similarity(correct_sent_vector, cand_sent_vector) # Cosine similarity between S(A) and S(Di)
        sent_similarities.append(sent_similarity)

    new_similarities = min_max_y(sent_similarities)
    
    for i, c in enumerate(cs):
        c["s2"] = 1-new_similarities[i]

    # POS match score s3
    origin_token = word_tokenize(sent)
    origin_token.remove("<")
    origin_token.remove(">")

    mask_index = origin_token.index("mask")

    correct_token = word_tokenize(correct_sent)
    correct_pos = nltk.pos_tag(correct_token)
    answer_pos = correct_pos[mask_index]    # POS of A

    for i, c in enumerate(cs):
        cand_sent_token = word_tokenize(cand_sents[i])
        cand_sent_pos = nltk.pos_tag(cand_sent_token)
        cand_pos = cand_sent_pos[mask_index]    # POS of Di

        if cand_pos[1] == answer_pos[1]:
            c["s3"] = 1.0
        else:
            c["s3"] = 0.0
        
    # Weighted final score
    cs_rank = list()
    for c in cs:
        fs = WEIGHT["s0"]*c["s0"] + WEIGHT["s1"]*c["s1"] + WEIGHT["s2"]*c["s2"] + WEIGHT["s3"]*c["s3"]
        cs_rank.append((c["word"], fs))

    # Rank by final score
    cs_rank.sort(key = lambda x: x[1], reverse=True)

    # Top K
    result = [d[0] for d in cs_rank[:TOP_K]]

    return result


# Cosine similarity
def similarity(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:  # Denominator can not be zero
        return 1
    else:
        return np.dot(v1, v2) / (n1 * n2)


# Min–max normalization
def min_max_y(raw_data):
    min_max_data = []
    
    # Min–max normalization
    for d in raw_data:
        try:
            min_max_data.append((d - min(raw_data)) / (max(raw_data) - min(raw_data)))
        except ZeroDivisionError:
            min_max_data.append(1)
                
    return min_max_data


if __name__ == "__main__":
    print("Distractor Generator (BART) Start!")
    main()
