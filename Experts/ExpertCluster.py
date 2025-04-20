import os 
import torch 
import torch.distributed as dist 
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import time
from datetime import timedelta


class ExpertCluster: 
    def __init__(self, modelPaths:list, backend="nccl"):
        self.backend = backend
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = os.environ["MASTER_PORT"]
        self.models = dict()
        self.tokenizers = dict()

        print(f"[{socket.gethostname()}][Rank {self.rank}] Starting ExpertCluster init.")
        self._init_process_group()

        if self.rank < len(modelPaths):
            print(f"[{socket.gethostname()}][Rank {self.rank}] Loading model {modelPaths[self.rank]}")
            self._load_model(modelPaths[self.rank])
        else:
            print(f"[{socket.gethostname()}][Rank {self.rank}] No model assigned, acting as coordinator.")

        dist.barrier()
        print(f"[{socket.gethostname()}][Rank {self.rank}] Passed initial barrier.")

    def _init_process_group(self):
        print(f"[{socket.gethostname()}][Rank {self.rank}] Initializing process group...")
        dist.init_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=30)
        )
        print(f"[{socket.gethostname()}][Rank {self.rank}] Process group initialized.")

    def _load_model(self, modelPath):
        self.tokenizers[self.rank] = AutoTokenizer.from_pretrained(modelPath)
        self.models[self.rank] = AutoModelForCausalLM.from_pretrained(modelPath).cuda()
        print(f"[{socket.gethostname()}][Rank {self.rank}] Model loaded.")

    def query(self, query, target):
        try:
            if self.rank == target:
                print(f"[{socket.gethostname()}][Rank {self.rank}] Received query directly.")
                return self.runInference(query)

            tensor = torch.tensor([ord(c) for c in query], dtype=torch.int).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=tensor, dst=target)
            print(f"[{socket.gethostname()}][Rank {self.rank}] Sent prompt to Node {target}")

            response_tensor = torch.empty(512, dtype=torch.int).cuda()
            dist.recv(tensor=response_tensor, src=target)
            print(f"[{socket.gethostname()}][Rank {self.rank}] Received response from Node {target}")

            response = "".join(chr(c) for c in response_tensor.cpu().tolist() if c != 0)
            return response
        except Exception as e:
            print(f"[{socket.gethostname()}][Rank {self.rank}] Error during query: {e}")
            return f"ERROR: {e}"

    def runInference(self, text):
        try:
            print(f"[{socket.gethostname()}][Rank {self.rank}] Running inference on: {text}")
            inputs = self.tokenizers[self.rank](text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                output = self.models[self.rank].generate(**inputs)
            result = self.tokenizers[self.rank].decode(output[0], skip_special_tokens=True)

            # Send back to master
            response_tensor = torch.zeros(512, dtype=torch.int).cuda()
            encoded = [ord(c) for c in result[:512]]
            response_tensor[:len(encoded)] = torch.tensor(encoded).cuda()
            torch.cuda.synchronize()
            dist.send(tensor=response_tensor, dst=0)
            print(f"[{socket.gethostname()}][Rank {self.rank}] Sent response back to master")
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
        print(f"[{socket.gethostname()}][Rank 0] Running tests...")
        responses = {}
        for target_rank in range(1, cluster.world_size):
            try:
                print(f"[{socket.gethostname()}][Rank 0] Querying Node {target_rank}...")
                responses[target_rank] = cluster.query(f"Test prompt for Node {target_rank}", target_rank)
                print(f"[{socket.gethostname()}][Rank 0] Response from Node {target_rank}: {responses[target_rank]}")
            except Exception as e:
                print(f"[{socket.gethostname()}][Rank 0] Failed to query Node {target_rank}: {e}")

    print(f"[{socket.gethostname()}][Rank {cluster.rank}] Waiting at final barrier...")
    dist.barrier()
    print(f"[{socket.gethostname()}][Rank {cluster.rank}] Final barrier passed.")

    dist.destroy_process_group()
    print(f"[{socket.gethostname()}][Rank {cluster.rank}] Exiting cleanly.")
