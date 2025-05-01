# %%

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
from autointerp import cache_activations
# from sparsify.sparsify import Sae
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import json
from types import SimpleNamespace # Import SimpleNamespace
from sae_lens import SAE
# from sae_lens import ActivationsStore
# import pandas as pd
# from tqdm import tqdm
import gc # Import gc for garbage collection
# instantiate an object to hold activations from a dataset

# a convenient way to instantiate an activation store is to use the from_sae method

# data = load_dataset("kh4dien/fineweb-sample", split="train[:25%]")

# model_id = "unsloth/Qwen2.5-Coder-32B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=t.bfloat16, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# path = "/workspace/qwen-saes-two/qwen-step-final/model.layers.31"
# sae = Sae.load_from_disk(path, device="cuda")

# path = "/workspace/qwen-saes-ft/qwen/layers.31"
# ssae = Sae.load_from_disk(path, device="cuda")

# set all torch tensors to cuda:1
# t.cuda.set_device(1)
data = load_dataset("cerebras/SlimPajama-627B", streaming=True)
# take first 1000 rows
data = data["train"].take(1000)
# get additional data from
additional_data = load_dataset("koyena/OpenR1-Math-220k-formatted", streaming=True)
additional_data = additional_data["train"].take(1000)
# get "text" column from data and "message_in_chat_template" column from additional_data
# Iterate to extract text data because IterableDataset doesn't support direct indexing
fineweb_texts = [example["text"] for example in data]
math_texts = [example["message_in_chat_template"] for example in additional_data]
# concatenate text and message_in_chat_template
# Combine the lists of texts
combined_texts = fineweb_texts + math_texts
print(len(combined_texts))
model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=t.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

SAE_LAYER = 15
RELEASE = "llama_scope_r1_distill"
SAE_ID = f"l{SAE_LAYER}r_400m_slimpajama_400m_openr1_math"
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"

sae, cfg_dict, sparsity = SAE.from_pretrained(
    # see other options in sae_lens/pretrained_saes.yaml
    release=RELEASE,
    sae_id=SAE_ID,
    device=DEVICE
)

sae.eval()

tokens = tokenizer(
    combined_texts,
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)

tokens = tokens["input_ids"]


# Revised encode function to return feature activations directly
# and include garbage collection / cache clearing
# def encode(x):
#     # Ensure SAE is on the same device as input tensor x
#     # Assuming x is already on DEVICE from cache_activations
#     flat_x = x.flatten(0, 1)
#     flat_resid = flat_x - sae.forward(flat_x)
#     print(flat_resid, flush=True)
#     B, S, _ = x.shape
#     resid = flat_resid.unflatten(0, (B, S))
#     # Encode the *residual* activations
#     resid_acts = sae.encode(resid)
#     del flat_x, flat_resid, resid # Explicitly delete intermediate tensors
#     if t.cuda.is_available():
#         t.cuda.empty_cache()
#     gc.collect()
#     print(resid_acts, flush=True)
#     return resid_acts


# def original_encode(x):
#     flat_x = x.flatten(0, 1)
#     flat_resid = flat_x - sae.simple_forward(flat_x)
#     print(flat_resid, flush=True)
#     B, S, _ = x.shape
#     resid = flat_resid.unflatten(0, (B, S))
#     return sae.simple_encode(resid)


cache = cache_activations(
    model=model.to(DEVICE),
    submodule_dict={"model.layers.15": sae.encode},
    tokens=tokens,
    batch_size=2,
    max_tokens=10_000_000,
)

# %%

save_dir = "/share/u/koyena/llama-8b-cache-sae-lens-with-slimpajama"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=50,
)
t.save(tokens, f"{save_dir}/tokens.pt")

