import os
import torch
import torch.distributed as dist
import time
import socket
import threading
import sys
from datetime import timedelta


# ======= Timeout Safeguard =======
def kill_after_timeout(timeout_seconds: int):
    def monitor():
        print(f"[{socket.gethostname()}] Watchdog started. Will kill process in {timeout_seconds} seconds.")
        time.sleep(timeout_seconds)
        print(f"[{socket.gethostname()}] Timeout reached. Exiting.")
        sys.exit(1)
    t = threading.Thread(target=monitor, daemon=True)
    t.start()


# ======= Debug Environment =======
def debug_env():
    print(f"[{socket.gethostname()}] ENV CHECKPOINT")
    print(f"[{socket.gethostname()}] ENV RANK         = {os.environ.get('RANK')}")
    print(f"[{socket.gethostname()}] ENV WORLD_SIZE   = {os.environ.get('WORLD_SIZE')}")
    print(f"[{socket.gethostname()}] ENV MASTER_ADDR  = {os.environ.get('MASTER_ADDR')}")
    print(f"[{socket.gethostname()}] ENV MASTER_PORT  = {os.environ.get('MASTER_PORT')}")


# ======= Safe Init Process Group =======
def safe_init_process_group(backend, init_method, rank, world_size):
    try:
        print(f"[{socket.gethostname()}][Rank {rank}] Initializing process group with {backend}...")
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30)
        )
        print(f"[{socket.gethostname()}][Rank {rank}] Process group initialized.")
    except Exception as e:
        print(f"[{socket.gethostname()}][Rank {rank}] Failed to initialize process group: {e}")
        sys.exit(1)


# ======= Main Entrypoint =======
def main():
    print(f"[{socket.gethostname()}] Script started.")
    kill_after_timeout(timeout_seconds=1200)

    debug_env()

    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
    except KeyError as e:
        print(f"[{socket.gethostname()}] Missing environment variable: {e}")
        sys.exit(1)

    print(f"[{socket.gethostname()}][Rank {rank}] Environment variables loaded.")

    init_method = f"tcp://{master_addr}:{master_port}"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    print(f"[{socket.gethostname()}][Rank {rank}] Backend: {backend}")
    print(f"[{socket.gethostname()}][Rank {rank}] Init method: {init_method}")

    safe_init_process_group(backend, init_method, rank, world_size)

    print(f"[{socket.gethostname()}][Rank {rank}] Allocating tensor...")
    tensor = torch.zeros(1).cuda() if torch.cuda.is_available() else torch.zeros(1)
    print(f"[{socket.gethostname()}][Rank {rank}] Tensor initialized with value {tensor.item()}")

    # Synchronize all processes before communication
    dist.barrier()

    if rank == 0:
        print(f"[{socket.gethostname()}][Rank 0] Adding 42 to tensor.")
        tensor += 42
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[{socket.gethostname()}][Rank 0] Sending tensor to rank 1...")
        dist.send(tensor=tensor, dst=1)
        print(f"[{socket.gethostname()}][Rank 0] Tensor sent.")
    elif rank == 1:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"[{socket.gethostname()}][Rank 1] Waiting to receive tensor from rank 0...")
        dist.recv(tensor=tensor, src=0)
        print(f"[{socket.gethostname()}][Rank 1] Received tensor with value: {tensor.item()}")

    # Final sync before shutdown
    dist.barrier()

    print(f"[{socket.gethostname()}][Rank {rank}] Destroying process group...")
    dist.destroy_process_group()
    print(f"[{socket.gethostname()}][Rank {rank}] Exiting cleanly.")


if __name__ == "__main__":
    main()
