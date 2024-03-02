# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter_v2 import GPT, Config
from lit_gpt.model import IntentionGPT
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load
# from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do llamas eat?",
    input: str = "",
    adapter_path: Path = Path("out/adapter_v2/alpaca/lit_model_adapter_finetuned.pth"),
    checkpoint_dir: Path = Path("/data/scz3286/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf/"),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    precision: Optional[str] = None,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-AdapterV2 model.
    See `finetune/adapter_v2.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        adapter_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/adapter_v2.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    # check_valid_checkpoint_dir(checkpoint_dir)

    config = Config.from_name(name="tiny-llama-1.1b")
    config.n_layer=2

    checkpoint_path = "/home/ubuntu/code/out/lit-tiny-llama-1.1b/step-00001000.pth"

    tokenizer = Tokenizer(checkpoint_dir)
    output = ""
    instructions = [
    """const std = @import("std"); const Allocator = std.mem.Allocator; const assert = std.debug.assert; const print = std.debug.print; const data = @embedFile("data/day08"); const EntriesList = std.ArrayList(Inst); const Op = enum { nop, acc, jmp, }; const Inst = struct { op: Op, val: isize, executed: bool = false, }; pub fn main() !void { var timer = try std.time.Timer.start(); var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator); const ally = &arena.allocator; var lines = std.mem.tokenize(data, "\r\n"); var entries = EntriesList.init(ally); try entries.ensureCapacity(400); while (lines.next()) |line| { if (line.len == 0) continue; var tok = std.mem.split(line, " "); var opname = tok.next().?; var int = try std.fmt.parseInt(isize, tok.next().?, 10); var op = if (std.mem.eql(u8, "nop", opname)) Op.nop else if (std.mem.eql(u8, "jmp", opname)) Op.jmp else Op.acc; try entries.append(.{ .op = op, .val = int, }); } var raw_items = entries.items; var items = try ally.alloc(Inst, raw_items.len); var result2: ?isize = null; for (raw_items) |item, i| { if (item.op == .nop or item.op == .jmp) { std.mem.copy(Inst, items, raw_items); if (item.op == .jmp) { items[i].op = .nop; } else { items[i].op = .jmp; } var pc: isize = 0; var acc: isize = 0; while (pc != @intCast(isize, items.len)) { var inst = &items[@intCast(usize, pc)]; if (inst.executed) break; inst.executed = true; switch(inst.op) { .jmp => { pc += inst.val; }, .acc => { acc += inst.val; pc += 1; }, .nop => { pc += 1; }, } } else { result2 = acc; break; } } } print("result: {}, time: {}\n", .{result2, timer.read()}); }
    """, 
    """const std = @import("std"); const Allocator = std.mem.Allocator; const assert = std.debug.assert; const print = std.debug.print; const data = @embedFile("data/day04.txt"); const Record = struct { byr: bool = false, iyr: bool = false, eyr: bool = false, hgt: bool = false, hcl: bool = false, ecl: bool = false, pid: bool = false, cid: bool = false, pub fn isValid(self: *const @This()) bool { return self.byr and self.iyr and self.eyr and self.hgt and self.hcl and self.ecl and self.pid; } }; pub fn main() !void { var lines = std.mem.split(data, "\n\n"); var numValid: usize = 0; while (lines.next()) |line| { var rec = Record{}; var toks = std.mem.tokenize(line, " \n"); while (toks.next()) |tok| { var colon = std.mem.indexOf(u8, tok, ":"); var tag = tok[0..colon.?]; var value = tok[colon.?+1..]; if (std.mem.eql(u8, "byr", tag)) { if (std.fmt.parseInt(u16, value, 10)) |ival| { if (ival >= 1920 and ival <= 2002) { rec.byr = true; } } else |err| {} } else if (std.mem.eql(u8, "iyr", tag)) { if (std.fmt.parseInt(u16, value, 10)) |ival| { if (ival >= 2010 and ival <= 2020) { rec.iyr = true; } } else |err| {} } else if (std.mem.eql(u8, "eyr", tag)) { if (std.fmt.parseInt(u16, value, 10)) |ival| { if (ival >= 2020 and ival <= 2030) { rec.eyr = true; } } else |err| {} } else if (std.mem.eql(u8, "hgt", tag)) { if (std.mem.endsWith(u8, value, "cm")) { if (std.fmt.parseInt(u16, value[0..value.len-2], 10)) |ival| { if (ival >= 150 and ival <= 193) { rec.hgt = true; } } else |err| {} } else if (std.mem.endsWith(u8, value, "in")) { if (std.fmt.parseInt(u16, value[0..value.len-2], 10)) |ival| { if (ival >= 59 and ival <= 76) { rec.hgt = true; } } else |err| {} } } else if (std.mem.eql(u8, "hcl", tag)) { if (value.len == 7 and value[0] == '#') { var valid = true; for (value[1..]) |char| { if (!((char >= '0' and char <= '9') or (char >= 'a' and char <= 'f'))) { valid = false; } } rec.hcl = valid; } } else if (std.mem.eql(u8, "ecl", tag)) { if ( std.mem.eql(u8, value, "amb") or std.mem.eql(u8, value, "blu") or std.mem.eql(u8, value, "brn") or std.mem.eql(u8, value, "gry") or std.mem.eql(u8, value, "grn") or std.mem.eql(u8, value, "hzl") or std.mem.eql(u8, value, "oth") ) { rec.ecl = true; } } else if (std.mem.eql(u8, "pid", tag)) { if (value.len == 9) { var valid = true; for (value[1..]) |char| { if (!(char >= '0' and char <= '9')) { valid = false; } } rec.pid = valid; } } else if (std.mem.eql(u8, "cid", tag)) { rec.cid = true; } else { print("Unknown tag: {}\n", .{tok}); } } numValid += @boolToInt(rec.isValid()); } print("Valid: {}\n", .{numValid}); }"""]
    for instruction in instructions:
        prompt = instruction

        # sample = {"instruction": prompt, "input": input}
        # prompt = generate_prompt(sample)
        encoded = tokenizer.encode(prompt, device=fabric.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        # fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
        t0 = time.perf_counter()
        with fabric.init_module(empty_init=True):
            model = IntentionGPT(config)
            # model.apply(partial(init_weights, n_layer=config.n_layer, n_embd=config.n_embd))

        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")

        model = torch.compile(model)
        model = fabric.setup(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=4e-4
        )
        optimizer = fabric.setup_optimizers(optimizer)

        state = {
            "model": model,
            "optimizer": optimizer,
            "train_dataloader": None,
            "hparams": None,
            "iter_num": 0,
            "step_count": 0,
        }
        fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
        with fabric.init_tensor():
            # set the max_seq_length to limit the memory usage to what we need
            model.max_seq_length = max_returned_tokens
            # enable the kv cache
            model.set_kv_cache(batch_size=1)
        model.eval()

        t0 = time.perf_counter()
        # checkpoint = lazy_load(checkpoint_path)
        # adapter_checkpoint = lazy_load(adapter_path)
        # checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
        # model = checkpoint["model"]
        model = fabric.setup(model)
        fabric.load(checkpoint_path, state)
        fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)


        L.seed_everything(1234)
        t0 = time.perf_counter()
        y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id, action_bias=0.1)
        t = time.perf_counter() - t0

        output = tokenizer.decode(y)
        # output = output.split("### Response:")[1].strip()
        fabric.print(output)

        tokens_generated = y.size(0) - prompt_length
        fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
        if fabric.device.type == "cuda":
            fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)








# def setup(
#     data_dir: Path = Path("data/alpaca"),
#     checkpoint_dir: Path = Path("/data/scz3286/lit-gpt/checkpoints/meta-llama/Llama-2-7b-hf/"),
#     out_dir: Path = Path("out/adapter_v2/alpaca"),
#     precision: Optional[str] = None,
#     quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
# ) -> None:
#     precision = precision or get_default_supported_precision(training=True)

#     plugins = None
#     if quantize is not None and quantize.startswith("bnb."):
#         if "mixed" in precision:
#             raise ValueError("Quantization and mixed precision is not supported.")
#         dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
#         plugins = BitsandbytesPrecision(quantize[4:], dtype)
#         precision = None

#     if devices > 1:
#         if quantize:
#             raise NotImplementedError(
#                 "Quantization is currently not supported for multi-GPU training. Please set devices=1 when using the"
#                 " --quantize flag."
#             )
#         strategy = FSDPStrategy(
#             auto_wrap_policy={Block},
#             activation_checkpointing_policy={Block},
#             state_dict_type="full",
#             limit_all_gathers=True,
#             cpu_offload=False,
#         )
#     else:
#         strategy = "auto"

#     logger = CSVLogger(out_dir.parent, out_dir.name, flush_logs_every_n_steps=log_interval)
#     fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger, plugins=plugins)
#     fabric.print(hparams)
#     fabric.launch(main, data_dir, checkpoint_dir, out_dir)


# def main(fabric: L.Fabric, data_dir: Path, checkpoint_dir: Path, out_dir: Path) -> None:
#     # check_valid_checkpoint_dir(checkpoint_dir)

#     fabric.seed_everything(1337)  # same seed for every process to init model (FSDP)

#     if fabric.global_rank == 0:
#         os.makedirs(out_dir, exist_ok=True)

#     # train_data = torch.load(data_dir / "train.pt")
#     # val_data = torch.load(data_dir / "test.pt")

#     config = Config.from_name(name=checkpoint_dir.name)
#     config.n_layer=2
#     checkpoint_path = checkpoint_dir / "lit_model.pth"
#     fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
#     with fabric.init_module(empty_init=(devices > 1)):
#         model = IntentionGPT(config)

#     fabric.print(f"Number of trainable parameters: {num_parameters(model, requires_grad=True):,}")
#     fabric.print(f"Number of non trainable parameters: {num_parameters(model, requires_grad=False):,}")
#     print("model attribute1", dir(model))
#     model = fabric.setup_module(model)
#     print("model attribute2", dir(model))

#     # trainable_params = [p for p in model.parameters() if p.requires_grad]

#     # strict=False because missing keys due to Adapter weights not contained in state dict
#     # load_checkpoint(fabric, model, checkpoint_path, strict=False)

#     fabric.seed_everything(1337 + fabric.global_rank)
#     tokenizer = Tokenizer(checkpoint_dir)
#     val_loss = validate(fabric, model, tokenizer)
#     fabric.print(f"val loss {val_loss.item():.4f}")
#     fabric.barrier()
#     if fabric.device.type == "cuda":
#         fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

# # the adapter "kv cache" cannot be initialized under `inference_mode`
# # @torch.no_grad()
# # def validate(fabric: L.Fabric, model: nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
# #     fabric.print("Validating ...")
# #     model.eval()

# #     losses = torch.zeros(max_iters, device=fabric.device)
# #     for k, val_data in enumerate(val_dataloader):
# #         if k >= max_iters:
# #             break
# #         input_ids = val_data[:, 0 : model.config.block_size].contiguous().long()
# #         targets = val_data[:, 1 : (model.config.block_size + 1)].contiguous().long()
# #         logits = model(input_ids)
# #         loss = chunked_cross_entropy(logits, targets)
# #         losses[k] = loss

# #     return losses.mean()

# @torch.no_grad()
# def validate(fabric: L.Fabric, model: IntentionGPT, tokenizer: Tokenizer) -> torch.Tensor:
#     fabric.print("Validating ...")
#     model.eval()

#     # produce an example: line 2066423 url:https://huggingface.co/datasets/bigcode/starcoderdata/viewer/default/train?p=2066422&row=206642212
#     instructions = ["const std = @import(\"std\"); const Allocator = std.mem.Allocator;", "const std = @import(\"std\");"]
#     # instruction = "Recommend a movie for me to watch during the weekend and explain the reason."
#     for instruction in instructions:
#         fabric.print(instruction)
#         # sample = {"instruction": instruction, "input": ""}
#         # prompt = generate_prompt(sample)
#         prompt = instruction
#         encoded = tokenizer.encode(prompt, device=fabric.device)
#         with fabric.init_tensor():
#             # do not set `max_seq_length=max_returned_token` because memory is not a concern here
#             model.set_kv_cache(batch_size=1)
#         output = generate(
#             model, encoded, max_returned_tokens=len(encoded) + eval_max_new_tokens, temperature=0.8, eos_id=tokenizer.eos_id, action_bias=0.1
#         )
#         model.clear_kv_cache()
#         output = tokenizer.decode(output)
#         fabric.print(output)


# def get_batch(
#     fabric: L.Fabric, data: List[Dict], longest_seq_ix: Optional[int] = None
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     ix = torch.randint(len(data), (micro_batch_size,))
#     if longest_seq_ix is not None:
#         # force the longest sample at the beginning so potential OOMs happen right away
#         ix[0] = longest_seq_ix

#     input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
#     labels = [data[i]["labels"].type(torch.int64) for i in ix]

#     # this could be `longest_seq_length` to have a fixed size for all batches
#     max_len = max(len(s) for s in input_ids)

#     def pad_right(x, pad_id):
#         # pad right based on the longest sequence
#         n = max_len - len(x)
#         return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

#     x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
#     y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

#     # Truncate if needed
#     if max_seq_length:
#         x = x[:, :max_seq_length]
#         y = y[:, :max_seq_length]

#     if fabric.device.type == "cuda" and x.device.type == "cpu":
#         x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
#     else:
#         x, y = fabric.to_device((x, y))
#     return x, y


# def get_lr_scheduler(optimizer, warmup_steps: int, max_steps: int):
#     # linear warmup followed by cosine annealing
#     scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
#     scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(max_steps - warmup_steps))
#     return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[warmup_steps])


# def get_longest_seq_length(data: List[Dict]) -> Tuple[int, int]:
#     # find out the minimum max_seq_length required during fine-tuning (saves memory!)
#     lengths = [len(d["input_ids"]) for d in data]
#     longest_seq_length = max(lengths)
#     longest_seq_ix = lengths.index(longest_seq_length)
#     return longest_seq_length, longest_seq_ix


# def save_adapter_v2_checkpoint(fabric: L.Fabric, model: torch.nn.Module, file_path: Path) -> None:
#     fabric.print(f"Saving adapter v2 weights to {str(file_path)!r}")
#     fabric.save(file_path, {"model": model}, filter={"model": adapter_filter})


# if __name__ == "__main__":
#     torch.set_float32_matmul_precision("high")

#     from jsonargparse import CLI

#     CLI(setup)
