from Aggregator.Aggregator import AggregationAgent
from router.Router import Router
from Experts.ExpertCluster import ExpertCluster

import os 
import torch

class HERD:
    def __init__(self, expert_model_paths, router_llm_path, router_zero_shot_path, aggregator_llm_path, aggregator_embedding_model_path, expert_domains):
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.expert_cluster = ExpertCluster(expert_model_paths)
        
        if self.rank == 0:
            self.router = Router(zero_shot_path=router_zero_shot_path, promptGenPath=router_llm_path, experts=expert_domains)
            self.aggregator = AggregationAgent(model_path=aggregator_llm_path, embedding_model=aggregator_embedding_model_path)

    def __call__(self, user_query):
        if self.rank == 0:
            expert_map = self.router.expertClassification(user_query)
            expert_prompts = self.router.buildExpertPrompts(expert_map, user_query)

            num_available_experts = self.world_size - 1
            expert_subset = list(expert_prompts.items())[:num_available_experts]

            rank_map = {rank: expert for rank, (expert, _) in zip(range(1, self.world_size), expert_subset)}
            prompts_by_rank = {rank: expert_prompts[expert] for rank, expert in rank_map.items()}

            responses = self.expert_cluster(prompts_by_rank)

            self.aggregator.build_index(list(responses.values()))
            final_answer = self.aggregator.query(user_query)

            return final_answer
        else:
            self.expert_cluster([])


if __name__ == "__main__":
    modelPaths = [
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--AI-MO--NuminaMath-7B-TIR/snapshots/cf2aaf3f706eef519a80523e21c655903203e984", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--Locutusque--OpenCerebrum-2.0-7B/snapshots/1fe44275e09e3d335fc214da06a7ac9be863341c"
    ]

    llmPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    zeroShotPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--facebook--bart-large-mnli/snapshots/d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"
    aggModelPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    embedPath = "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

    expertDomains = {
        "Biology": [
            "biology", "medicine", "anatomy", "cells", "organelles", "genetics", "gene expression", "DNA", "RNA",
            "protein synthesis", "immunology", "evolution", "microbiology", "biotechnology", "neuroscience",
            "enzymes", "metabolism", "pathology", "physiology", "biochemistry", "virology", "bacteria", "viruses",
            "cell biology", "molecular biology", "bioinformatics", "pharmacology", "genomics", "epigenetics"
        ],

        "Computer Science": [
            "computer science", "algorithms", "data structures", "Python", "Java", "C++", "C#", "Rust", "Go", 
            "programming", "machine learning", "deep learning", "AI", "artificial intelligence", "neural networks", 
            "NLP", "transformers", "databases", "SQL", "NoSQL", "APIs", "computer vision", "software engineering", 
            "compilers", "operating systems", "parallel computing", "distributed systems", "cloud computing", 
            "Docker", "Kubernetes", "Git", "DevOps", "system design", "networking", "cybersecurity"
        ],

        "Math": [
            "mathematics", "calculus", "linear algebra", "probability", "statistics", "discrete math", 
            "number theory", "set theory", "combinatorics", "algebra", "differential equations", "integration", 
            "derivatives", "geometry", "topology", "vector calculus", "tensor", "group theory", "graph theory",
            "optimization", "matrices", "math proofs", "logic", "real analysis", "complex analysis", "trigonometry"
        ],

        "Physics": [
            "physics", "classical mechanics", "thermodynamics", "quantum mechanics", "relativity", 
            "electromagnetism", "waves", "optics", "particle physics", "nuclear physics", "kinematics", 
            "dynamics", "conservation of energy", "conservation of momentum", "force", "mass", "acceleration", 
            "velocity", "fields", "gravitational waves", "quantum fields", "spin", "angular momentum", 
            "wave-particle duality", "superposition", "interference", "photons", "electrons", "atomic physics"
        ]
    }

    herd = HERD(
        expert_model_paths=modelPaths,
        router_llm_path=llmPath,
        router_zero_shot_path=zeroShotPath,
        aggregator_llm_path=aggModelPath,
        aggregator_embedding_model_path=embedPath,
        expert_domains=expertDomains
    )

    if herd.rank == 0:
        prompt = "Explain how energy conservation applies to orbital mechanics using math and physics."
        print(herd(prompt))
    else:
        herd([])  
