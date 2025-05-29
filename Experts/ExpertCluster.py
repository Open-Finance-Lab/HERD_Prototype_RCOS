import os 
import torch 
import torch.distributed as dist 
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
from datetime import timedelta


class ExpertCluster: 
    def __init__(self, modelPaths: list, expert_to_rank: dict, backend="nccl"):
        self.backend = backend
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.models = {}
        self.tokenizers = {}
        self.expert_to_rank = expert_to_rank
        self.rank_to_expert = {v: k for k, v in expert_to_rank.items()}

        dist.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=60)
        )

        if self.rank > 0 and self.rank in self.rank_to_expert:
            expert_name = self.rank_to_expert[self.rank]
            model_path_index = self.rank - 1

            if model_path_index < len(modelPaths):
                print(f"[Rank {self.rank}] Loading model for expert '{expert_name}' from {modelPaths[model_path_index]}")
                self.tokenizers[expert_name] = AutoTokenizer.from_pretrained(modelPaths[model_path_index])
                self.models[expert_name] = AutoModelForCausalLM.from_pretrained(modelPaths[model_path_index]).cuda()
                print(f"[Rank {self.rank}] Finished loading model for expert '{expert_name}'")
            else:
                print(f"[Rank {self.rank}] No model path available for this rank.")
        else:
            print(f"[Rank {self.rank}] Acting as master node (no model will be loaded).")

        dist.barrier()

    def query(self, query, target_rank):
        try:
            if self.rank == target_rank:
                return self.runInference(query)

            tensor = torch.tensor([ord(c) for c in query], dtype=torch.int).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=tensor, dst=target_rank)

            response_tensor = torch.empty(512, dtype=torch.int).cuda()
            dist.recv(tensor=response_tensor, src=target_rank)

            response = "".join(chr(c) for c in response_tensor.cpu().tolist() if c != 0)
            return response
        except Exception as e:
            return f"ERROR: {e}"

    def runInference(self, text):
        try:
            expert_name = self.rank_to_expert[self.rank]
            tokenizer = self.tokenizers[expert_name]
            model = self.models[expert_name]

            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=128)
            result = tokenizer.decode(output[0], skip_special_tokens=True)

            response_tensor = torch.zeros(512, dtype=torch.int).cuda()
            encoded = [ord(c) for c in result[:512]]
            response_tensor[:len(encoded)] = torch.tensor(encoded).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=response_tensor, dst=0)
        except Exception as e:
            print(f"[{socket.gethostname()}][Rank {self.rank}] Inference failed: {e}")
        return None

    def __call__(self, prompts_by_expert: dict):
        if self.rank == 0:
            responses = dict()
            for expert, prompt in prompts_by_expert.items():
                target_rank = self.expert_to_rank.get(expert)
                if target_rank is None:
                    print(f"[Rank 0] No rank assigned to expert: {expert}")
                    continue
                print(f"[Rank 0] Sending prompt for expert {expert} to rank {target_rank}")
                try:
                    responses[expert] = self.query(prompt, target_rank)
                except Exception as e:
                    print(f"Error querying expert '{expert}' at rank {target_rank}: {e}")
            return responses
        else:
            if self.rank not in self.rank_to_expert:
                print(f"[Rank {self.rank}] No expert assigned to this rank.")
                return

            try:
                queryTensor = torch.empty(512, dtype=torch.int).cuda()
                print(f"[Rank {self.rank}] Waiting for prompt...")
                dist.recv(tensor=queryTensor, src=0)
                receivedPrompt = "".join(chr(c) for c in queryTensor.cpu().tolist() if c != 0)
                print(f"[Rank {self.rank}] Received prompt: {receivedPrompt[:50]}...")
                self.runInference(receivedPrompt)
            except Exception as e:
                print(f"[Rank {self.rank}] Error handling incoming prompt: {e}")


if __name__ == "__main__":
    modelPaths = [
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--AI-MO--NuminaMath-7B-TIR/snapshots/cf2aaf3f706eef519a80523e21c655903203e984", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--Locutusque--OpenCerebrum-2.0-7B/snapshots/1fe44275e09e3d335fc214da06a7ac9be863341c"
    ]

    expert_to_rank = {
        "Biology": 1,
        "Math": 2,
        "Computer Science": 3
    }

    cluster = ExpertCluster(modelPaths, expert_to_rank)

    if cluster.rank == 0:
        prompts_by_expert = {
            "Biology": "Explain CRISPR and its medical implications.",
            "Math": "What is the Riemann Hypothesis?",
            "Computer Science": "How do transformers work in NLP?"
        }
        responses = cluster(prompts_by_expert)
        for expert, response in responses.items():
            print(f"[Final Response from {expert}]: {response}")
    else:
        cluster({})

    dist.barrier()
    dist.destroy_process_group()
