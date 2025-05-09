from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import math

class Router():
    def __init__(self, zero_shot_path:str, promptGenPath:str, experts:dict):
        self.classifier = pipeline("zero-shot-classification", model=zero_shot_path)
        self.expertList = experts
        self.tokenizer = AutoTokenizer.from_pretrained(promptGenPath)
        self.model = AutoModelForCausalLM.from_pretrained(promptGenPath, device_map="auto", torch_dtype="auto")

        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=False
        )

    def expertClassification(self, prompt:str):
        labels = [item for sublist in self.expertList.values() for item in sublist]
        relevantExperts = dict()
        base_threshold = 0.15
        adjusted_threshold = base_threshold / math.log(len(labels) + 1)
        result = self.classifier(prompt, candidate_labels=labels)
        print(result["scores"]) #debug print
        print(result["labels"])
        for index, score in enumerate(result["scores"]):
            if (score >= adjusted_threshold):
                print(result["labels"][index])
                matchingKeys = [k for k, v in self.expertList.items() if (result["labels"])[index] in v]
                if matchingKeys[0] not in relevantExperts.keys():
                    relevantExperts[matchingKeys[0]] = [(result["labels"])[index]]
                else:
                    relevantExperts[matchingKeys[0]].append((result["labels"])[index])
        return relevantExperts
    
    def buildExpertPrompts(self, expertDict: dict, originalPrompt: str):
        expertPrompts = dict()
        for expert in expertDict.keys():
            promptTemplate = f"""You are a prompt generator for a routing mechanism in a multi-expert system.
            Original Prompt: {originalPrompt}
            Expert Domain: {expert}
            Relevant Domain Keywords/Topics: {', '.join(expertDict[expert])}

            Rewrite the prompt so that it is focused specifically on the {expert} domain.
            The prompt should allow the {expert} expert to focus only on {expert}-specific instructions from the original prompt.
            """
            result = self.llm(promptTemplate)[0]["generated_text"]
            formattedResult = result.replace(promptTemplate, "").strip()
            expertPrompts[expert] = formattedResult
        return expertPrompts

if __name__ == "__main__":
    llmPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    zeroShotPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    prompt = "Can you make me a python program that uses computer vision to identify cancer cells in an electron-microscope image? Make sure to highlight biological processes"

    experts = {"Biology" : ["biology", "medicine", "biological processes", "anatomy", "cells", "cell structure", "biotechnology", "genes", "gene editing"],
               "Computer Science" : ["Computer Science", "Python", "C++", "Java", "C#", "Programming"], 
               "Physics" : ["Physics"],
               "Math" : ["Math"]
               }
    
    routingAgent = Router(zero_shot_path=zeroShotPath, promptGenPath=llmPath, experts=experts)
    relevantExperts = routingAgent.expertClassification(prompt)
    print(relevantExperts)
    expertPrompts = routingAgent.buildExpertPrompts(relevantExperts, prompt)
    print(expertPrompts)