import ExpertCluster
import torch.distributed as dist
import os


def expertTest():
    """
    This test function lays out how to use the expert cluster object in an external script. 
    """
    modelPaths = [
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--BioMistral--BioMistral-7B/snapshots/9a11e1ffa817c211cbb52ee1fb312dc6b61b40a5", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--AI-MO--NuminaMath-7B-TIR/snapshots/cf2aaf3f706eef519a80523e21c655903203e984", 
        "/gpfs/u/home/ARUS/ARUSgrsm/scratch/HFModels/models--Locutusque--OpenCerebrum-2.0-7B/snapshots/1fe44275e09e3d335fc214da06a7ac9be863341c"
    ]

    prompts = [None, "what is 4+4", "what is 8+8"]

    ExpCluster = ExpertCluster.ExpertCluster(modelPaths)

    if ExpCluster.rank == 0:
        responses = ExpCluster(prompts)
    else: 
        ExpCluster(prompts)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    expertTest()