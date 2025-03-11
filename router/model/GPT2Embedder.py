import torch, spacy
from transformers import GPT2Tokenizer, GPT2Model

class GPT2Embedder:
    """
    Extracts tokenized text and contextual embeddings from GPT-2.
    """
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2Model.from_pretrained("gpt2")

    def get_tokens(self, text):
        """
        Tokenizes input text using GPT-2 tokenizer.
        Returns tokenized text.
        """
        return self.tokenizer.tokenize(text)

    def get_embeddings(self, text):
        """
        Extracts token embeddings from GPT-2.
        Returns a tensor of shape (1, num_tokens, embedding_dim).
        """
        encoded_input = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input)
        return output.last_hidden_state
    
    def get_attention_weights(self, text):
        """
        Extracts token attention weights from a GPT-2 model.
        """
        encoded_input = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded_input, output_attentions=True)
        attention = output.attentions[-1]  # Last layer attention weights
        return attention.mean(dim=1)  # Average across heads

    def extract_important_pos(self, text):
        """
        Extracts nouns and verbs as potential important tokens.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        important_tokens = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        return important_tokens