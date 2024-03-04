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
from tqdm import tqdm

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
global_batch_size = 2048
learning_rate = 2e-4
micro_batch_size = 64
max_tokens = int(20e9)  # 3 trillion  # 20 Billion
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

def mail(name="", msg=""):
    import smtplib
    from email.mime.text import MIMEText
    from email.utils import formataddr
    my_sender='aid0214@163.com'    # 发件人邮箱账号
    my_pass = 'MCJHFHOXVODVRAVV'              # 发件人邮箱密码
    my_user='jiacx@lamda.nju.edu.cn'      # 收件人邮箱账号，我这边发送给自己

    ret=True
    try:
        # msg=MIMEText('填写邮件内容','plain','utf-8')      #此处为仅填写文本数据
        msg=MIMEText(msg,'html','utf-8')      #需要发送html数据的时候用这种形式
        msg['From']=formataddr(["FromRunoob",my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To']=formataddr(["FK",my_user])              # 括号里的对应收件人邮箱昵称、收件人邮箱账号
        msg['Subject']=name                # 邮件的主题，也可以说是标题
 
        server=smtplib.SMTP_SSL("smtp.163.com", 465)  # 发件人邮箱中的SMTP服务器
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender,[my_user,],msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret=False
    return ret


def setup(resume: Union[bool, Path] = True):
    logger = choose_logger(logger_name, name=name, resume=resume)

    strategy = FSDPStrategy(auto_wrap_policy={Block}, cpu_offload=False, limit_all_gathers=True, activation_checkpointing_policy=None, state_dict_type="full", sharding_strategy="FULL_SHARD")
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
    if test:
        config.n_layer = 2

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

    # validate(fabric, model, val_dataloader, max_iters=2)  # sanity check
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
        save_info = {
                "model": model,
                "optimizer": optimizer,
                # "train_dataloader": train_dataloader,
                "hparams": hparams,
                "iter_num": 0,
                "step_count": 0,
            }
        fabric.save(checkpoint_path, save_info)
    
    best_loss = 100000.
    grad_before = None
    grad_after = None
    
    action_max = None
    action_min = None
    for train_data in tqdm(train_iterator):
        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"], warmup_iters, max_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        state["iter_num"] += 1
        iter_t0 = time.perf_counter()
        input_ids = train_data[:, 0 : model.config.block_size].contiguous().long()
        
        try:
            problem = 0
            is_accumulating = state["iter_num"] % gradient_accumulation_iters != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                with torch.no_grad():
                    action = model(input_ids, train_mode=True, action_only=True)
                
                action_max_batch = action.reshape(-1, action.shape[-1]).max(dim=0, keepdim=False)[0]
                action_min_batch = action.reshape(-1, action.shape[-1]).min(dim=0, keepdim=False)[0]
                if action_max is None:
                    action_max = action_max_batch
                else:
                    action_max = torch.max(action_max, action_max_batch)
                    
                if action_min is None:
                    action_min = action_min_batch
                else:
                    action_min = torch.max(action_min, action_min_batch)
                
                if state["iter_num"] % 100 == 0:
                    print({"max": action_max.detach().cpu().numpy(), "min": action_min.detach().cpu().numpy()})
                    print({"max": action_max.detach().cpu().numpy(), "min": action_min.detach().cpu().numpy()}, file=open("action_bound_iter.txt", "w"))
                
        except:
            print(input_ids.shape, problem, file=open("test.txt", "w"))
            mail(name=name, msg="experiments killed at {}".format(str(state["iter_num"])))
            assert 0
    print({"max": action_max.detach().cpu().numpy(), "min": action_min.detach().cpu().numpy()}, file=open("action_bound.txt", "w"))


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

    train_datasets = [
        StreamingDataset(
            input_dir="/data/scz3286/lit-gpt/data/slimpajama/train",
            item_loader=TokensLoader(block_size=effective_block_size),
            shuffle=True,
            drop_last=True,
        ),
        StreamingDataset(
            input_dir="/data/scz3286/lit-gpt/data/starcoder",
            item_loader=TokensLoader(block_size=effective_block_size),
            shuffle=True,
            drop_last=True,
        ),
    ]
    # train_datasets = StreamingDataset(
    #     # input_dir="/data/wangpy/Research/data/starcoder",
    #     input_dir="/data/scz3286/lit-gpt/data/starcoder",
    #     item_loader=TokensLoader(block_size=effective_block_size),
    #     shuffle=True,
    #     drop_last=True,
    # )

    # Mix SlimPajama data and Starcoder data with these proportions:
    weights = (0.693584, 0.306416)
    combined_dataset = CombinedStreamingDataset(datasets=train_datasets, seed=42, weights=weights)
    train_dataloader = StreamingDataLoader(
        combined_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers, drop_last=True
    )

    val_dataset = StreamingDataset(
        # input_dir="/data/wangpy/Research/data/starcoder_eval",
        input_dir="/data/scz3286/lit-gpt/data/slimpajama/validation",
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
