import torch
from transformers import pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

class SentimentRLAgent:
    def __init__(self, model_name="gpt2"):
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.tokenizer = pipeline("text-generation", model=model_name)
    
    def generate_response(self, prompt):
        response = self.tokenizer(prompt, max_length=50, do_sample=True)[0]["generated_text"]
        return response