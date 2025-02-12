import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    """
    Uses a transformer-based model to analyze token-level sentiment.
    """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def analyze_sentiment(self, tokens):
        """
        Computes sentiment scores for each token.
        Returns a dictionary with token-wise sentiment probabilities.
        """
        sentiment_scores = {}
        for token in tokens:
            encoded_token = self.tokenizer(token, return_tensors="pt")
            with torch.no_grad():
                sentiment_output = self.model(**encoded_token)
            score = torch.softmax(sentiment_output.logits, dim=1)[0]  # [negative_prob, positive_prob]
            sentiment_scores[token] = score.numpy().tolist()
        return sentiment_scores