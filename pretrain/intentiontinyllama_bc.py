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

import lightning as L
import torch
import torch.nn as nn
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

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

"""
(micro=2, global 512, hybrid-shard, activation-chpt=None, ): TFLOPs: 765.85, 47.2 GB, 707.48 ms
(micro=3, global 768, hybrid-shard, activation-chpt=None, ): TFLOPs: 1148.78, 61.0 GB, 973.72 ms * 2/3
(micro=4, global 512, hybrid-shard, activation-chpt=None, ): TFLOPs: 1531.71, 75.4 GB, 1238.57 ms * 1/2
(micro=4, global 512, hybrid-shard, activation-chpt=None, ): TFLOPs: 1531.71, 75.4 GB, 1238.57 ms * 1/2
(micro=4, global 512, full-shard, activation-chpt=None, cpu-off=True): TFLOPs: 1531.71, 74.4 GB, 1238.50 ms * 1/2
(micro=4, global 512, fhybrid-shard, activation-chpt={BLOCK}, ): TFLOPs: 1531.71, 27.3 GB, 1814.85 ms * 1/2
(micro=8, global 512, fhybrid-shard, activation-chpt={BLOCK}, ): TFLOPs: 3063.41, 39.3 GB, 3663.52 ms * 1/4
(micro=16, global 512, fhybrid-shard, activation-chpt={BLOCK}, ): TFLOPs: 6126.82, 58.3 GB, 6580.31 ms * 1/8
(micro=24, global 768, fhybrid-shard, activation-chpt={BLOCK}, ): TFLOPs: 9190.23, 68.5 GB, 9724.71 ms * 1/12
(micro=4, global 512, hybrid-shard, activation-chpt=None, gather-lim=True): TFLOPs: 1531.71, 75.4 GB, 1238.57 ms * 1/2
(micro=4, global 512, hybrid-shard, activation-chpt=None, gather-lim=False): TFLOPs: 1531.71, 80.1 GB, 1236.20 ms * 1/2
"""

beta = 1.5
hidden_dim = 64
# System settings
model_name = "tiny-llama-1.1b"
test = False
name = "lit-tiny-llama-1.1b-beta={}-hidden-dim={}".format(beta, hidden_dim)
if test:
    name = "lit-tiny-llama-1.1b-beta-test"
out_dir = Path(os.getenv("LIGHTNING_ARTIFACTS_DIR", "out")) / name
logger_name = "tensorboard"
devices = torch.cuda.device_count() or 1

# Hyperparameters
global_batch_size = 512
learning_rate = 4e-4
micro_batch_size = 2
max_tokens = int(3e9)  # 3 trillion  # 20 Billion
warmup_steps = 2000
log_step_interval = 1
eval_iters = 100
save_step_interval = 500
eval_step_interval = 500

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


def setup(resume: Union[bool, Path] = False):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False, limit_all_gathers=True, activation_checkpointing_policy=None, state_dict_type="full", sharding_strategy="HYBRID_SHARD")
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

    train_dataloader, val_dataloader = create_dataloaders(batch_size=micro_batch_size, block_size=config.block_size)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)  # same seed for every process to init model (FSDP)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = IntentionGPT(config, hidden_dim=hidden_dim)
        model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters: {num_parameters(model):,}")

    model = torch.compile(model)
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {
        "model": model,
        "optimizer": optimizer,
        "train_dataloader": train_dataloader,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=(lambda p: int(p.name.split("-")[1])))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
    throughput = ThroughputMonitor(fabric, window_size=5)

    with torch.device("meta"):
        meta_model = IntentionGPT(model.config, hidden_dim=hidden_dim)
        x = torch.randint(0, 1, (micro_batch_size, meta_model.config.block_size))
        model_fwd = lambda: meta_model(x)
        model_loss = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, model_fwd, model_loss)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    max_tokens_per_device = max_tokens // fabric.world_size
    tokens_per_iter = micro_batch_size * model.config.block_size
    max_iters = max_tokens_per_device // tokens_per_iter
    initial_iter = state["iter_num"]
    train_iterator = CycleIterator(train_dataloader)

    running_loss = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    running_loss_enc = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    running_loss_dec = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    # running_loss_bc = RunningMean(window=gradient_accumulation_iters, sync_on_compute=False).to(fabric.device)
    fabric.barrier()
    total_t0 = time.perf_counter()
    
    if test:
        checkpoint_path = out_dir / f"step-{0:08d}.pth"
        fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
        fabric.save(checkpoint_path, state)
    
    best_loss = 100000.
    for train_data in train_iterator:
        if state["iter_num"] >= max_iters:
            break

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous().long()
        targets = train_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
        
        try:
            problem = 0
            is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits, info = model(input_ids, train_mode=True)
                enc_loss = chunked_kld(info['mean'], info['logvar'])
                dec_loss = chunked_cross_entropy(logits, targets)
                # bc_loss = chunked_bc(info['mean'], info['logvar'], info['mean_bc'], info['logvar_bc'])
                loss = beta * enc_loss + dec_loss #+ bc_loss
                fabric.backward(loss / gradient_accumulation_iters)
                entropy = compute_entropy(logits.detach())
                
                problem = 1
                
                # bc_logits, bc_info = model(input_ids.detach(), train_mode=True, action_copy=True)
                # bc_enc_loss = chunked_kld(bc_info['mean'], bc_info['logvar'])
                # bc_dec_loss = chunked_cross_entropy(bc_logits, targets.detach())
                # # bc_loss = chunked_bc(info['mean'], info['logvar'], info['mean_bc'], info['logvar_bc'])
                # bc_loss = beta * bc_enc_loss + bc_dec_loss #+ bc_loss
                # fabric.backward(bc_loss / gradient_accumulation_iters)
                # bc_entropy = compute_entropy(bc_logits.detach())
        except:
            print(input_ids.shape, problem, file=open("test.txt", "w"))
            assert 0

        running_loss.update(loss.detach())
        running_loss_enc.update(enc_loss.detach())
        running_loss_dec.update(dec_loss.detach())
        # running_loss_bc.update(bc_loss.detach())

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if state["iter_num"] % log_iter_interval == 0:
            loss = running_loss.compute().item()  # expensive device-to-host synchronization
            enc_loss = running_loss_enc.compute().item()  # expensive device-to-host synchronization
            dec_loss = running_loss_dec.compute().item()  # expensive device-to-host synchronization
            # bc_loss = running_loss_bc.compute().item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=(t1 - total_t0),
                flops=(measured_flops * log_iter_interval),
                batches=state["iter_num"],
                samples=(state["iter_num"] * micro_batch_size),
                lengths=(state["iter_num"] * micro_batch_size * model.config.block_size),
            )
            metrics = {
                "loss": dec_loss,
                "loss_enc": enc_loss,
                "loss_dec": dec_loss,
                # "loss_bc": bc_loss,
                "loss_total": loss,
                "value/output_entropy": entropy.item(),
                "value/ent_mean": info['entropy_mean'].item(),
                "value/ent_std": info['entropy_std'].item(),
                "value/ent_max": info['entropy_max'].item(),
                "value/ent_min": info['entropy_min'].item(),
                "value/mu_mean": info['mean_mean'].item(),
                "value/mu_std": info['mean_std'].item(),
                "value/mu_max": info['mean_max'].item(),
                "value/mu_min": info['mean_min'].item(),
                "value/std_mean": info['std_mean'].item(),
                "value/std_std": info['std_std'].item(),
                "value/std_max": info['std_max'].item(),
                "value/std_min": info['std_min'].item(),
                
                # "bc_value/output_entropy": bc_entropy.item(),
                # "bc_value/ent_mean": bc_info['entropy_mean'].item(),
                # "bc_value/ent_std": bc_info['entropy_std'].item(),
                # "bc_value/ent_max": bc_info['entropy_max'].item(),
                # "bc_value/ent_min": bc_info['entropy_min'].item(),
                # "bc_value/mu_mean": bc_info['mean_mean'].item(),
                # "bc_value/mu_std": bc_info['mean_std'].item(),
                # "bc_value/mu_max": bc_info['mean_max'].item(),
                # "bc_value/mu_min": bc_info['mean_min'].item(),
                # "bc_value/std_mean": bc_info['std_mean'].item(),
                # "bc_value/std_std": bc_info['std_std'].item(),
                # "bc_value/std_max": bc_info['std_max'].item(),
                # "bc_value/std_min": bc_info['std_min'].item(),
                
                "iter": state["iter_num"],
                "step": state["step_count"],
                "epoch": train_iterator.epoch,
                "iter_time": t1 - iter_t0,
                "remaining_time": (
                    (t1 - total_t0) / (state["iter_num"] - initial_iter) * (max_iters - state["iter_num"])
                ),
                "tokens": state["iter_num"] * micro_batch_size * model.config.block_size,
                "total_tokens": state["iter_num"] * micro_batch_size * model.config.block_size * fabric.world_size,
                "learning_rate": lr,
            }

            fabric.print(
                f"iter {metrics['iter']} | step {metrics['step']}: loss {metrics['loss']:.4f}, loss_dec {metrics['loss_dec']:.4f}, mu {metrics['value/mu_mean']:4f}, std {metrics['value/std_mean']:4f}, iter time:"
                f" {metrics['iter_time'] * 1000:.2f} ms{' (optimizer.step),' if not is_accumulating else ','}"
                f" remaining time: {metrics['remaining_time'] / 3600 / 24:.2f} days"
            )

            throughput_metrics = throughput.compute()
            metrics.update(throughput_metrics)
            fabric.log_dict(metrics, step=state["iter_num"])

        if val_dataloader is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
            val_loss = val_loss.item()
            td = time.perf_counter() - t0
            
            if val_loss < best_loss:
                best_loss = val_loss

            fabric.print(f"iter {state['iter_num']}: val loss {val_loss:.4f}, val time: {td * 1000:.2f} ms")
            metrics = {"val_loss": val_loss, "val_ppl": math.exp(val_loss)}
            fabric.log_dict(metrics, step=state["iter_num"])
            fabric.barrier()

        if not is_accumulating and state["step_count"] % save_step_interval == 0:
            checkpoint_path = out_dir / f"step-{state['step_count']:08d}.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(max_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= max_iters:
            break
        input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
        targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
        logits = model(input_ids)
        loss = chunked_cross_entropy(logits, targets)
        losses[k] = loss

    model.train()
    return losses.mean()


def create_dataloaders(batch_size: int, block_size: int, num_workers: int = 8) -> Tuple[DataLoader, DataLoader]:
    from lightning.data import CombinedStreamingDataset, StreamingDataLoader, StreamingDataset
    from lightning.data.streaming.item_loader import TokensLoader

    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1

    # train_datasets = [
    #     StreamingDataset(
    #         input_dir="data/slimpajama/train",
    #         item_loader=TokensLoader(block_size=effective_block_size),
    #         shuffle=True,
    #         drop_last=True,
    #     ),
    #     StreamingDataset(
    #         input_dir="data/starcoder",
    #         item_loader=TokensLoader(block_size=effective_block_size),
    #         shuffle=True,
    #         drop_last=True,
    #     ),
    # ]
    train_datasets = StreamingDataset(
        # input_dir="/data/wangpy/Research/data/starcoder",
        input_dir="/data/scz3286/lit-gpt/data/starcoder",
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        drop_last=True,
    )

    # Mix SlimPajama data and Starcoder data with these proportions:
    # weights = (0.693584, 0.306416)
    # combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights)
    train_dataloader = StreamingDataLoader(
        train_datasets, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )

    val_dataset = StreamingDataset(
        # input_dir="/data/wangpy/Research/data/starcoder_eval",
        input_dir="/data/scz3286/lit-gpt/data/starcoder_eval",
        item_loader=TokensLoader(block_size=effective_block_size),
        shuffle=True,
        # Consider setting to False, but we would lose some samples due to truncation when world size > 1
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def init_weights(module: nn.Module, n_layer: int, n_embd: int):
    # Follows GPT-NeoX: https://arxiv.org/abs/2204.06745
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / n_embd))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    for name, param in module.named_parameters():
        if name == "proj.weight" and isinstance(module, (LLaMAMLP, CausalSelfAttention)):
            nn.init.normal_(param, mean=0.0, std=(1 / math.sqrt(n_embd) / n_layer))


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
