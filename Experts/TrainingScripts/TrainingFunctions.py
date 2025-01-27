from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset

def loadModel(modelPath : str):
    model = AutoModelForCausalLM.from_pretrained(modelPath, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    return model, tokenizer

def PrepDataSet(tokenizer : AutoTokenizer, datasetPath : str, textCol : str = "text"):
    dataset = load_dataset(datasetPath)

    tokenizedData = dataset.map(
        lambda examples: tokenizer(examples[textCol], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    return tokenizedData

