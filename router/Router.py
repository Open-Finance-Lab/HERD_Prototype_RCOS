#Strategy: Use label families instead of just general labels, like make groups of labels for a expert. 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class Router():
    def __init__(self, zero_shot_path:str, experts:dict):
        self.classifier = pipeline("zero-shot-classification", model=zero_shot_path)
        self.expertList = experts

    def expertClassification(self, prompt:str):
        labels = [item for sublist in self.expertList.values() for item in sublist]
        relevantExperts = list()

        result = self.classifier(prompt, candidate_labels=labels)
        print(result["scores"]) #debug print

        for index, score in enumerate(result["scores"]):
            if (score >= 0.2):
                print(result["labels"][index])
                matchingKeys = [k for k, v in self.expertList.items() if (result["labels"])[index] in v]
                relevantExperts.append(matchingKeys[0])
                #add section to turn relevent experts into a dict so we can group key words that activated the expert
                #with the expert for prompt building
        return relevantExperts


if __name__ == "__main__":

    experts = {"Biology" : ["biology", "medicine", "biological processes", "anatomy", "cells", "cell structure", "biotechnology", "genes", "gene editing"],
               "Computer Science" : ["Computer Science", "Python", "C++", "Java", "C#", "Programming"], 
               "Physics" : ["Physics"],
               "Math" : ["Math"]
               }
    
    routingAgent = Router("/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce", experts)
    relevantExperts = routingAgent.expertClassification("Can you make me a python program that uses computer vision to identify cancer cells in an electron-microscope image?")
    print(relevantExperts)