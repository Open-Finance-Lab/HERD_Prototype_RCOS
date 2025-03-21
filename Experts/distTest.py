import os
import torch
import torch.distributed as dist
import time
import socket

def debug_env():
    print(f"[{socket.gethostname()}] ENV RANK = {os.environ.get('RANK')}")
    print(f"[{socket.gethostname()}] ENV WORLD_SIZE = {os.environ.get('WORLD_SIZE')}")
    print(f"[{socket.gethostname()}] ENV MASTER_ADDR = {os.environ.get('MASTER_ADDR')}")
    print(f"[{socket.gethostname()}] ENV MASTER_PORT = {os.environ.get('MASTER_PORT')}")

def safe_init_process_group(backend, init_method, rank, world_size):
    try:
        print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.timedelta(seconds=30)
        )
        print(f"[Rank {rank}] Process group initialized.")
    except Exception as e:
        print(f"[Rank {rank}] Failed to initialize process group: {e}")
        exit(1)

def main():
    debug_env()

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    init_method = f"tcp://{master_addr}:{master_port}"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    safe_init_process_group(backend, init_method, rank, world_size)

    # Tensor communication test
    tensor = torch.zeros(1).cuda() if torch.cuda.is_available() else torch.zeros(1)

    if rank == 0:
        print("[Rank 0] Preparing tensor to send...")
        tensor += 42
        print("[Rank 0] Sending tensor to rank 1...")
        dist.send(tensor=tensor, dst=1)
        print("[Rank 0] Tensor sent.")
    elif rank == 1:
        print("[Rank 1] Waiting to receive tensor...")
        dist.recv(tensor=tensor, src=0)
        print(f"[Rank 1] Received tensor: {tensor.item()}")

    dist.destroy_process_group()
    print(f"[Rank {rank}] Exiting cleanly.")

if __name__ == "__main__":
    main()
