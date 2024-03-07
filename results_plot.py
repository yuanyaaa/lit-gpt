import os
import collections
from glob import glob
from pprint import pprint

import math
import yaml
import json
# import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
RESULT = collections.namedtuple("result", "tb config dir")

sns.set_style(style="darkgrid", rc=None)
name_dict = {
    "intention-gpt": "lit-tiny-llama-1.1b-beta={}-hidden-dim={}{}",
    "baseline-gpt": "tiny-llama-1.1b",
}
PATH = "out/{}/logs/tensorboard/version_{}"
config = {
    "per_device_training_batch_size": 512 // 8,
    "world_size": 2048,
}

def make_template(x_name, *y_names):
    template = {}
    template[x_name] = []
    for y_name in y_names:
        template[y_name] = []
    return template

def load_result(load_dir):
    print(load_dir)
    def read_tb_file(filename):
        ea = event_accumulator.EventAccumulator(
            filename, size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
        )
        ea.Reload()

        kvs = dict()
        for key in ea.scalars.Keys():
            events = ea.scalars.Items(key)
            if key not in kvs:
                kvs[key] = []
            for e in events:
                kvs[key].append((e.step, e.wall_time, e.value))
        return kvs

    tb_path = glob(os.path.join(load_dir, "events.out.tfevents.*"))
    if len(tb_path) == 0:
        return None
    tb = read_tb_file(tb_path[0])

    result = RESULT(tb=tb, config=config, dir=load_dir)
    return result


def plot_evaluation_result(all_result, save_dir="analysis"):
    name_map = {
        "Decoder Loss": "loss_dec",
        "Encoder Loss (log)": "loss_enc",
        "Output Entropy": "value/output_entropy",
        "Action Mean": "value/mu_mean",
        "Action Std (log)": "value/std_mean",
        "Action Entropy": "value/ent_mean",
        "Gradient (log)": "grad/grad_before",
        # "Validation Loss": "val_loss",
    }
    lens = len(name_map.keys())

    def custom_round(x, bins=6):
        precision = 1 / bins
        integer_part = math.floor(x)
        decimal_part = x - integer_part
        bin_number = math.floor(decimal_part / precision)
        rounded_decimal_part = bin_number * precision
        rounded_number = integer_part + rounded_decimal_part
        return rounded_number

    variables = list(name_map.keys())
    templates = []
    names = []
    for result, name in all_result:
        config = result.config
        tb = result.tb
        tmp_template = make_template("Iteration", *variables)
        start_time = None
        bs = config["per_device_training_batch_size"] * config.get("world_size", 4)
        # additional results
        other_keys = list(name_map.keys())
        for key in other_keys:
            tmp_template["Iteration"].extend([int(x[0]) for x in tb[name_map[key]]])
            tmp_template[key].extend([x[2] if "log" not in key else math.log(x[2]) for x in tb[name_map[key]]])
        templates.append(tmp_template)
        names.append(name)
    
    
    plt.figure(figsize=(30, 12))
    for num, key in enumerate(list(name_map.keys())):
        plt.subplot(2, 4, num+1)
        for template in templates:
            plt.plot(np.arange(len(template[key]))*16, template[key], linewidth=2)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(key, fontsize=32)
    
    plt.legend(names, fontsize=25, loc=(1.1, 0.5))
    save_dir = os.path.join(save_dir, "test")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "pic.png"))


def plot():
    exp_list = [
        # ("intention-gpt", 1.5, 64, 1, "beta-vae"), 
        # ("intention-gpt", 5.0, 8, 1, "beta-vae (CE)"), 
        # ("intention-gpt", 5.0, 8, 2, "beta-vae"),
        # ("intention-gpt", 5.0, 8, 3, "beta-vae (KL)"),
        ("intention-gpt", 1.5, 4, 0, "-emb"),
        ("intention-gpt", 1.5, 4, 2, "-no_noise"),
        ("intention-gpt", 1.5, 4, 6, ""),
    ]
    res_list = []
    for exp in exp_list:
        result = load_result(PATH.format(name_dict[exp[0]].format(exp[1], exp[2], exp[4]), exp[3]))
        res_list.append([result, "beta={}, dim={}, k={}".format(exp[1], exp[2], exp[4])])
    plot_evaluation_result(res_list)

plot()