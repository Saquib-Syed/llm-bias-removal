#%%
%cd ~/llm-bias-removal
%load_ext autoreload
%autoreload 2

from transformers import AutoTokenizer, AutoModelForCausalLM
from evals.evals import batched_ave_logit_diff
from dataset.load_dataset import load_data, get_toks, get_tok_pos

import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
model = HookedTransformer.from_pretrained(
    "google/gemma-2b",
    default_padding_side="left"
)
tokenizer = model.tokenizer
model.eval()
model.to(device)

#%%
from dataset.custom_dataset import PairedInstructionDataset
import json

with open(f'dataset/gender_bias_ds.json', 'r') as f:
    data = json.load(f)

def tokenize_instructions(tokenizer, instructions):
    try:
        return torch.tensor(tokenizer(instructions, padding=True).input_ids)
    except:
        print(instructions)

dataset = PairedInstructionDataset(
    N=1500,
    instruction_templates=data['instruction_templates'],
    harmful_substitution_map=data['male_sub_map'],
    harmless_substitution_map=data['female_sub_map'],
    tokenizer=tokenizer,
    tokenize_instructions=tokenize_instructions
)
_, he_tok, she_tok, him_tok, his_tok, her_tok = tokenizer.encode(" he she him his her")
# %%
from evals.evals import logit_diff_on_gender

with torch.set_grad_enabled(False):
    male = batched_ave_logit_diff(
            model,
            dataset.harmful_dataset.toks,
            toks_a=torch.tensor([he_tok]),
            toks_b=torch.tensor([she_tok]),
            batch_size=500,
            do_mean=False
        )
    female = batched_ave_logit_diff(
            model,
            dataset.harmless_dataset.toks,
            toks_a=torch.tensor([he_tok]),
            toks_b=torch.tensor([she_tok]),
            batch_size=500,
            do_mean=False
        )

#%%
with open('results/prompted_2.json', 'w') as f:
    json.dump({'male_jobs': male.tolist(), 'female_jobs': female.tolist()}, f)
# %%
