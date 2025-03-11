import spacy

def extract_important_tokens(text):
    # Process the text
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    important_tokens = []
    
    for token in doc:
        # Keep key POS tags (nouns, verbs, adjectives, adverbs)
        if token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
            important_tokens.append((token.text, token.pos_, token.dep_))

    # Extract named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return {"tokens": important_tokens, "named_entities": named_entities}

if __name__ == "__main__":
    text = "Can you give me code to reverse a linked list?"

    important_tokens = extract_important_tokens(text)

    print (important_tokens)