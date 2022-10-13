from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "./models/CSG/BERT_CLOTH_model"
PRETRAIN_MODEL = "bert-base-uncased"
HUGGINGFACE_MODEL = ""

print(f"Load CSG model at {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL)
csg_model = AutoModelForMaskedLM.from_pretrained(MODEL)