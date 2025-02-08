import torch
from transformers import BertForSequenceClassification, BertTokenizer

class SentimentRewardModel:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def compute_reward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        sentiment_score = torch.softmax(outputs.logits, dim=1)
        reward = sentiment_score[:, 4].item()  # Assuming 5-star sentiment (0-4 scale)
        return reward