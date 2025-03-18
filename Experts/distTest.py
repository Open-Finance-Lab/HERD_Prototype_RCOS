import os
import torch
import torch.distributed as dist

def main():
    # SLURM environment variables for PyTorch
    rank = int(os.environ['RANK'])  # Process ID
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPUs, Gloo for CPUs
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    )

    # Tensor communication test
    tensor = torch.zeros(1).cuda()  # Place on GPU if available

    if rank == 0:
        tensor += 42  # Modify tensor on rank 0
        dist.send(tensor=tensor, dst=1)  # Send tensor to rank 1
        print(f"Node {rank} sent: {tensor.item()}")
    elif rank == 1:
        dist.recv(tensor=tensor, src=0)  # Receive tensor from rank 0
        print(f"Node {rank} received: {tensor.item()}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
