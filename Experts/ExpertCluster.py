#|----------------------------------------------|
#|This script is made to support SLURM currently|
#|----------------------------------------------|

import os 
import torch 
import torch.distributed as dist 
from transformers import AutoModelForCausalLM, AutoTokenizer

class ExpertCluster: 
#|---------Initialization----------|
    def __init__(self, modelPaths:list, backend="nccl"):
        """
        modelPaths : list of the paths to each expert, 1 to 1 mapping for index and node rank. 
        """
        self.backend = backend
        self.rank = int(os.environ.get("SLURM_PROCID", 0))
        self.world_size = int(os.environ.get("SLURM_NTASKS", 1))
        self.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        self.models = dict()
        self.tokenizers = dict()

        if self.rank == 0:
            self._init_process_group()

        if self.rank < len(modelPaths):
            self._load_model(modelPaths[self.rank])

    def _init_process_group(self):
        """
        Init the process group for the expert cluster
        """
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            backend=self.backend,
            rake=self.rank,
            world_size=self.world_size
        )
    
    def _load_model(self, modelPath):
        """
        Puts model and tokenizer into dictionaries
        """
        self.tokenizers[self.rank] = AutoTokenizer.from_pretrained(modelPath)
        self.models[self.rank] = AutoModelForCausalLM.from_pretrained(modelPath).cuda()

#|----------Querying----------|
    def query(self, query, target):
        if self.rank == target: #if the node we run script on is the same as the expert
            return self.runInference(query)
        
        tensor = torch.tensor([ord(c) for c in query], dtype=torch.int).cuda()
        dist.send(tensor=tensor, dst=target)
        print(f"Master: Sent prompt to Node {target}")

        # Receive the response from the target node
        response_tensor = torch.empty(512, dtype=torch.int).cuda()  # Assume max response length
        dist.recv(tensor=response_tensor, src=target)

        response = "".join(chr(c) for c in response_tensor.cpu().tolist() if c != 0)
        return response

    def runInference(self, text):
        inputs = self.tokenizers[self.rank](text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self.models[self.rank].generate(**inputs)
        return self.tokenizers[self.rank].decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":

    modelPaths = ["/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5", 
                  "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--AI-MO--NuminaMath-7B-TIR/snapshots/cf2aaf3f706eef519a80523e21c655903203e984", 
                  "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--Locutusque--OpenCerebrum-2.0-7B/snapshots/1fe44275e09e3d335fc214da06a7ac9be863341c"]

    cluster = ExpertCluster(modelPaths)

    if (cluster.rank == 0):
        print("Running Tests")

        responses = {}
        for target_rank in range(1, cluster.world_size):
            responses[target_rank] = cluster.query(f"Test prompt for Node {target_rank}", target_rank)

        for node, response in responses.items():
            print(f"Response from Node {node}: {response}")


"""
LOGS: 
Currently, the process group is not working, but the model loading between nodes is working fine, so I need to work on inter-node communication now. 
"""