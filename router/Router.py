from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Router():
    def __init__(self, zero_shot_path:str, experts:list):
        self.classifier = pipeline("zero-shot-classification", model=zero_shot_path)
        self.expertList = experts

    def expertClassification(self, prompt:str):
        result = self.classifier(prompt, candidate_labels=self.expertList)
        relevantExperts = list()

        for index, score in enumerate(result["scores"]):
            if (score >= 0.2):
                relevantExperts.append((result["labels"])[index])
        
        return relevantExperts


if __name__ == "__main__":

    # classifier = pipeline("zero-shot-classification", model="/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce")

    # prompt = "The anti-diuretic hormone controls the regulation of urea in the human body."

    # labels = ["Biology", "Anatomy", "Physics", "Math"]

    # result = classifier(prompt, candidate_labels=labels)

    # print(result["labels"])
    # print(result["scores"])

    experts = ["Biology", "Computer Science", "Physics", "Math"]
    routingAgent = Router("/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce", experts)
    relevantExperts = routingAgent.expertClassification("Can you make me a python program that uses computer vision to identify cancer cells in an electron-microscope image?")
    print(relevantExperts)