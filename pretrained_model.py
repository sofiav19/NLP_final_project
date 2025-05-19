# download_model_once.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name, force_download=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, force_download=True)
print("Download completed successfully.")
