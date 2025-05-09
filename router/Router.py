from transformers import pipeline
#Zero-Shot Test

classifier = pipeline("zero-shot-classification", model="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce")

prompt = "Explain how neural networks are used in image classification and object detection."

labels = ["computer vision", "natural language processing", "optimization", "hardware acceleration"]

result = classifier(prompt, candidate_labels=labels)

print(result["labels"])
print(result["scores"])