# CDGP

Code for Findings of EMNLP 2022 short paper **"CDGP: Automatic Cloze Distractor Generation based on Pre-trained Language Model"**.

* [Paper](https://github.com/AndyChiangSH/CDGP/blob/main/paper/CDGP%20Automatic%20Cloze%20Distractor%20Generation%20based%20on%20Pre-trained%20Language%20Model.pdf)
* [Demo page](https://cdgp-demo.nlpnchu.org/)

## 🗂 Structure

* `paper/`: "CDGP: Automatic Cloze Distractor Generation based on Pre-trained Language Model"
* `models/`: models in CDGP
    * `CSG/`: the models as Candidate Set Generator
    * `DS/`: the models as Distractor Selector
* `datasets/`: datasets for fine-tuneing and testing
    * `CLOTH.zip`: CLOTH dataset
    * `DGen.zip`: DGen dataset
* `fine-tune/`: code for fine-tuning
* `test/`: code for testing
    * `dis_generator(BERT).py`: distractors generator based on BERT
    * `dis_generator(SciBERT).py`: distractors generator based on SciBERT
    * `dis_generator(RoBERTa).py`: distractors generator based on RoBERTa
    * `dis_generator(BART).py`: distractors generator based on BART
    * `dis_evaluator.py`: distractors evaluator
    * `results/`: results of distractors generator
    * `evaluations/`: evaluations of distractors evaluator
* `demo.ipynb`: code for CDGP demo

## ❤ Models

Models are available at Hugging Face.

### Candidate Set Generator (CSG)

Its input are stem and answer, and output is candidate set of distractors.

| Models      | CLOTH                                                                               | DGen                                                                             |
| ----------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **BERT**    | [cdgp-csg-bert-cloth](https://huggingface.co/AndyChiang/cdgp-csg-bert-cloth)        | [cdgp-csg-bert-dgen](https://huggingface.co/AndyChiang/cdgp-csg-bert-dgen)       |
| **SciBERT** | [cdgp-csg-scibert-cloth](https://huggingface.co/AndyChiang/cdgp-csg-scibert-cloth)  | [cdgp-csg-scibert-dgen](https://huggingface.co/AndyChiang/cdgp-csg-scibert-dgen) |
| **RoBERTa** | [cdgp-csg-roberta-cloth](https://huggingface.co/AndyChiang/cdgp-csg-roberta-cloth) | [cdgp-csg-roberta-dgen](https://huggingface.co/AndyChiang/cdgp-csg-roberta-dgen) |
| **BART**    | [cdgp-csg-bart-cloth](https://huggingface.co/AndyChiang/cdgp-csg-bart-cloth)        | [cdgp-csg-bart-dgen](https://huggingface.co/AndyChiang/cdgp-csg-bart-dgen)       |

### Distractor Selector (DS)

Its input are stem, answer and candidate set of distractors, and output are top 3 distractors.

**fastText**: [cdgp-ds-fasttext](https://huggingface.co/AndyChiang/cdgp-ds-fasttext)

## 📚 Datasets

Datasets are available at Hugging Face and GitHub.

### CLOTH

[CLOTH](https://www.cs.cmu.edu/~glai1/data/cloth/) is a dataset which is a collection of nearly 100,000 cloze questions from middle school and high school English exams. The detail of CLOTH dataset is shown below.

| Number of questions | Train | Valid | Test  |
| ------------------- | ----- | ----- | ----- |
| **Middle school**   | 22056 | 3273  | 3198  |
| **High school**     | 54794 | 7794  | 8318  |
| **Total**           | 76850 | 11067 | 11516 |

You can download CLOTH dataset from [Hugging Face](https://huggingface.co/datasets/AndyChiang/cloth) or [GitHub](https://github.com/AndyChiangSH/CDGP/blob/main/datasets/CLOTH.zip).

### DGen

[DGen](https://github.com/DRSY/DGen) is a cloze questions dataset which covers multiple domains including science, vocabulary, common sense and trivia. It is compiled from a wide variety of datasets including SciQ, MCQL, AI2 Science Questions, etc. The detail of DGen dataset is shown below.

| DGen dataset            | Train | Valid | Test | Total |
| ----------------------- | ----- | ----- | ---- | ----- |
| **Number of questions** | 2321  | 300   | 259  | 2880  |

You can download CLOTH dataset from [Hugging Face](https://huggingface.co/datasets/AndyChiang/dgen) or [GitHub](https://github.com/AndyChiangSH/CDGP/blob/main/datasets/DGen.zip).

## 📝 Evaluations

The evaluations of these model as a Candidate Set Generator in CDGP is shown as follows:

### CLOTH

| Models                                                                             | P@1   | F1@3  | F1@10 | MRR   | NDCG@10 |
| ---------------------------------------------------------------------------------- | ----- | ----- | ----- | ----- | ------- |
| [**cdgp-csg-bert-cloth**](https://huggingface.co/AndyChiang/cdgp-csg-bert-cloth)       | 18.50 | 13.80 | 15.37 | 29.96 | 37.82   |
| [**cdgp-csg-scibert-cloth**](https://huggingface.co/AndyChiang/cdgp-csg-scibert-cloth) | 8.10  | 9.13  | 12.22 | 19.53 | 28.76   |
| [**cdgp-csg-roberta-cloth**](https://huggingface.co/AndyChiang/cdgp-csg-roberta-cloth) | 10.50 | 9.83  | 10.25 | 20.42 | 28.17   |
| [**cdgp-csg-bart-cloth**](https://huggingface.co/AndyChiang/cdgp-csg-bart-cloth)       | 14.20 | 11.07 | 11.37 | 24.29 | 31.74   |

### DGen

| Models                                                                           | P@1   | F1@3  | MRR   | NDCG@10 |
| -------------------------------------------------------------------------------- | ----- | ----- | ----- | ------- |
| [**cdgp-csg-bert-dgen**](https://huggingface.co/AndyChiang/cdgp-csg-bert-dgen)       | 10.81 | 7.72  | 18.15 | 24.47   |
| [**cdgp-csg-scibert-dgen**](https://huggingface.co/AndyChiang/cdgp-csg-scibert-dgen) | 13.13 | 12.23 | 25.12 | 34.17   |
| [**cdgp-csg-roberta-dgen**](https://huggingface.co/AndyChiang/cdgp-csg-roberta-dgen) | 13.13 | 9.65  | 19.34 | 24.52   |
| [**cdgp-csg-bart-dgen**](https://huggingface.co/AndyChiang/cdgp-csg-bart-dgen)       | 8.49  | 8.24  | 16.01 | 22.66   |

## 💡 How to use?

### Setup environment

1. Clone or download this repo.

```
git clone https://github.com/AndyChiangSH/CDGP.git
```

2. Move into this repo.

```
cd ./CDGP/
```

3. Setup a virtual environment.

```
python -m venv CDGP-env
```

> Python version: 3.8.8

4. Pip install the required packages.

```
pip install -r requirements.txt
```

### Fine-tune

Our model is fine-tuned on [Colab](https://colab.research.google.com/), so you can upload [these Jupyter Notebook](https://github.com/AndyChiangSH/CDGP/tree/main/fine-tune) to Colab and run it by yourself!

### Test

We are testing in local, so we need to download the datasets and models.

1. Unzip the CLOTH or DGen datasets in `/datasets/`.
2. CSG models will download from Hugging Face when you run the code, so you don't have to do anything!
3. If you want to use your own CSG model, you can put it in the new directory `/models/CSG/`.
4. However, you have to download the DS models by yourself.
5. Then, move the DS models into the new directory `/models/DS/`.
6. Run `/test/dis_generator(BERT).py` to generate the distractors based on BERT.
7. Run `/test/dis_generator(SciBERT).py` to generate the distractors based on SciBERT.
8. Run `/test/dis_generator(RoBERTa).py` to generate the distractors based on RoBERTa.
9. Run `/test/dis_generator(BART).py` to generate the distractors based on BART.
10. Check the generating results as `.json` files in `/test/results/`.
11. Run `/test/dis_evaluator.py` to evaluate the generating results.
12. Check the evaluations as `.csv` file in `/test/evaluations/`.

## 📌 Citation

```
None
```

## 😀 Author

* Shang-Hsuan Chiang ([@AndyChiangSH](https://github.com/AndyChiangSH))
* Ssu-Cheng Wang ([@shiro-wang](https://github.com/shiro-wang))
* Yao-Chung Fan
