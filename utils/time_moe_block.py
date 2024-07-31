
import torch
import time
from tqdm.auto import tqdm


"""
For timing MoE blocks
Pass list of initialized classes and config with settings, returns average time for each tested block

cfg = config

blocks = [
    #SentenceMultiTaskMoeBlock1(cfg, EsmExpert),
    #SentenceMultiTaskMoeBlock2(cfg, EsmExpert),
]
"""

class config:
    seq_len = 128
    batch_size = 8
    hidden_dim = 768
    runs = 100
    intermediate_dim = 2048


def time_block(cfg, blocks):
    times = []
    for i, block in enumerate(blocks):
        block.to(cfg.device)
        start = time.time()
        for _ in tqdm(range(cfg.runs), desc=f'{i}'):
            hidden_states = torch.rand(cfg.batch_size, cfg.seq_len, cfg.hidden_dim).to(cfg.device)
            _, _ = block(hidden_states)
        end = time.time()
        total_time = (end - start) / cfg.runs
        times.append(total_time)
    return times


if __name__ == '__main__':
    from models.moe_blocks import SentenceTokenTypeMoeBlock, TokenMultiTaskMoeBlock
    from models.MLP import VanillaMLP

    pass ### TODO