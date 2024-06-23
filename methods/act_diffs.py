#%%
%cd ../
from transformers import AutoTokenizer, AutoModelForCausalLM
from evals.evals import batched_ave_logit_diff
from dataset.load_dataset import load_data, get_toks, get_tok_pos

import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

model = HookedTransformer.from_pretrained(
    "google/gemma-2b",
)
tokenizer = model.tokenizer
model.eval()
model.to(device)

#%% Logit diff across model components

#%% PCA of activations of the most important components

#%% Act diffs

#%% Probing

#%% Ablations

#%% Orthogonalization
