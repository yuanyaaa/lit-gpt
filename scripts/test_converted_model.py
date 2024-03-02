import torch
from transformers import AutoModel


state_dict = torch.load("./out/hf_model/converted.pth")
# model = AutoModel.from_pretrained(
#     "output_path/", local_files_only=True, state_dict=state_dict
# )
model = AutoModel.from_pretrained("TinyLlama/TinyLlama-1.1B-step-50K-105b", state_dict=state_dict)