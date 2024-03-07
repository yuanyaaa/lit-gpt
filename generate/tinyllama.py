# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""
This script is adapted from TinyLlama:
https://github.com/jzhang38/TinyLlama/blob/main/pretrain/tinyllama.py
"""

import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Union

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate, generate_with_actions
import lightning as L
import torch
import torch.nn as nn
from lit_gpt import Tokenizer
from lightning.fabric.loggers import CSVLogger, TensorBoardLogger
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics.aggregation import RunningMean

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.model import IntentionGPT, GPT, Block, CausalSelfAttention, Config, LLaMAMLP
from lit_gpt.utils import CycleIterator, chunked_cross_entropy, chunked_kld, chunked_bc, compute_entropy, num_parameters

import torch._dynamo
torch._dynamo.config.suppress_errors = True

beta = 0.5
hidden_dim = 32
# System settings
model_name = "tiny-llama-1.1b"
name = "lit-tiny-llama-1.1b-beta"
name = "tiny-llama-1.1b"
out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
logger_name = "tensorboard"
devices = torch.cuda.device_count() or 1

# Hyperparameters
global_batch_size = 512
learning_rate = 4e-4
micro_batch_size = 2
max_tokens = int(2e9)  # 3 trillion  # 20 Billion
warmup_steps = 2000
log_step_interval = 1
eval_iters = 100
save_step_interval = 500
eval_step_interval = 100

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
min_lr = 4e-5

batch_size = global_batch_size // devices
gradient_accumulation_iters = batch_size // micro_batch_size
assert gradient_accumulation_iters > 0
warmup_iters = warmup_steps * gradient_accumulation_iters
log_iter_interval = log_step_interval * gradient_accumulation_iters


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}


def setup(resume: Union[bool, Path] = True):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
    fabric = L.Fabric(devices=devices, strategy=strategy, precision="bf16-mixed", loggers=[logger])
    fabric.launch()

    fabric.print(hparams)
    if logger_name in ("tensorboard", "wandb"):
        fabric.logger.log_hyperparams(hparams)

    main(fabric, resume)


def main(fabric, resume):
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    # train_dataloader, val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    # train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        # model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    # )
    # optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        # "optimizer": optimizer,
        # "train_dataloader": train_dataloader,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)
        model.set_kv_cache(batch_size=1, device=fabric.device)

    max_new_tokens = 50
    checkpoint_dir: Path = Path("/data/scz3286/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf/")
    tokenizer = Tokenizer(checkpoint_dir)
    
    actions = []
    
    sentence = """In buildings containing flats or in housing developments where there are private areas used in common and not owned by specific properties,"""
    instructions = [
    """In buildings containing flats or in housing developments where there are private areas used in common and not owned by specific properties, the cost of maintaining those areas needs to be recovered. With flats these charges are called 'service charges' and in housing developments they are called 'rent charges'. The maintenance is usually organised by a third party, such as a landlord or a management company. Usually an estimate as to the charge is given at the beginning of a pre-defined year and a payment on account made. At the end of the year, accounts are produced and balancing credits or additional payments made. The service charge costs can be significant and you need to consider this when purchasing. You need to consider the history of payments and future payments. Buildings with flat roofs or lifts can produce particularly high service charges. You can access any of our property lawyers at any of our offices spread throughout Surrey, Hampshire and Greater London including Kingston upon Thames, Bordon, Cheam, Canary Wharf, Leatherhead, Raynes Park, Surbiton, Tolworth or Walton on Thames."""]
    # sentence = """const std = @import("std"); const Allocator = std.mem.Allocator; const assert = std.debug.assert; const print = std.debug.print; const data = @embedFile("data/day0"""
    # instructions = [
    # """const std = @import("std"); const Allocator = std.mem.Allocator; const assert = std.debug.assert; const print = std.debug.print; const data = @embedFile("data/day08"); const EntriesList = std.ArrayList(Inst);""", 
    # """const std = @import("std"); const Allocator = std.mem.Allocator; const assert = std.debug.assert; const print = std.debug.print; const data = @embedFile("data/day04.txt"); const Record = struct"""]
    for instruction in instructions:
        encoded = tokenizer.encode(sentence, device=fabric.device).contiguous().long()
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens
        y = generate(model, encoded, max_returned_tokens, temperature=0.8, top_k=200, eos_id=tokenizer.eos_id, action_bias=0.0)
        t = time.perf_counter() - t0
        output = tokenizer.decode(y)
        # output = output.split("### Response:")[1].strip()
        fabric.print(output)
        print(output)

def choose_logger(logger_name: str, name: str, resume: Union[bool, Path], *args, **kwargs):
    if logger_name == "csv":
        return CSVLogger(root_dir=(out_dir / "logs"), name="csv", *args, **kwargs)
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), name="tensorboard", *args, **kwargs)
    if logger_name == "wandb":
        return WandbLogger(project="tinyllama", name=name, resume=(resume is not False), *args, **kwargs)
    raise ValueError(f"`logger={logger_name}` is not a valid option.")

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
