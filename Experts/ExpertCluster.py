import os 
import torch 
import torch.distributed as dist 
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
from datetime import timedelta


class ExpertCluster: 
    def __init__(self, modelPaths:list, backend="nccl"):
        self.backend = backend
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.models = {}
        self.tokenizers = {}

        dist.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=30)
        )

        if self.rank < len(modelPaths):
            self.tokenizers[self.rank] = AutoTokenizer.from_pretrained(modelPaths[self.rank])
            self.models[self.rank] = AutoModelForCausalLM.from_pretrained(modelPaths[self.rank]).cuda()

        dist.barrier()

    def query(self, query, target):
        try:
            if self.rank == target:
                return self.runInference(query)

            tensor = torch.tensor([ord(c) for c in query], dtype=torch.int).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=tensor, dst=target)

            response_tensor = torch.empty(512, dtype=torch.int).cuda()
            dist.recv(tensor=response_tensor, src=target)

            response = "".join(chr(c) for c in response_tensor.cpu().tolist() if c != 0)
            return response
        except Exception as e:
            return f"ERROR: {e}"

    def runInference(self, text):
        try:
            inputs = self.tokenizers[self.rank](text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = self.models[self.rank].generate(**inputs)
            result = self.tokenizers[self.rank].decode(output[0], skip_special_tokens=True)

            response_tensor = torch.zeros(512, dtype=torch.int).cuda()
            encoded = [ord(c) for c in result[:512]]
            response_tensor[:len(encoded)] = torch.tensor(encoded).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=response_tensor, dst=0)
        except Exception as e:
            print(f"[{socket.gethostname()}][Rank {self.rank}] Inference failed: {e}")
        return None


if __name__ == "__main__":
    modelPaths = [
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--AI-MO--NuminaMath-7B-TIR/snapshots/cf2aaf3f706eef519a80523e21c655903203e984", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--Locutusque--OpenCerebrum-2.0-7B/snapshots/1fe44275e09e3d335fc214da06a7ac9be863341c"
    ]

    cluster = ExpertCluster(modelPaths)

    if cluster.rank == 0:
        responses = {}
        for target_rank in range(1, cluster.world_size):
            responses[target_rank] = cluster.query(
                f"Write a short sentence explaining why Node {target_rank} is important.",
                target_rank
            )
            print(f"Response from Node {target_rank}: {responses[target_rank]}")
    else:
        try:
            query_tensor = torch.empty(512, dtype=torch.int).cuda()
            dist.recv(tensor=query_tensor, src=0)
            received_query = "".join(chr(c) for c in query_tensor.cpu().tolist() if c != 0)
            cluster.runInference(received_query)
        except Exception as e:
            print(f"[Rank {cluster.rank}] Error: {e}")

    dist.barrier()
    dist.destroy_process_group()
