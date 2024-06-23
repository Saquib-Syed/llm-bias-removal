#%%
%cd ../
from transformers import AutoTokenizer, AutoModelForCausalLM
from evals.evals import batched_ave_logit_diff
from dataset.load_dataset import load_data, get_toks, get_tok_pos

import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "google/gemma-2b",
)
tokenizer = model.tokenizer
model.eval()
model.to('mps:0')

#%%
df = load_data()
toks, occ_toks = get_toks(df, tokenizer)
tok_pos = get_tok_pos(toks, occ_toks, tokenizer)
toks = torch.tensor(toks)
_, he_tok, she_tok, his_tok, her_tok = tokenizer.encode(" he she his her")
pronoun_pos = torch.tensor(
    [
        pronoun[1] # last idx of pronoun
        for pronoun in tok_pos["{pronoun}"]
    ]
)
#%%
with torch.set_grad_enabled(False):
    b = batched_ave_logit_diff(
        model, 
        toks[:10], 
        pronoun_pos,
        torch.tensor([he_tok, his_tok]),
        torch.tensor([she_tok, her_tok]),
        batch_size=2
    )
print(b)
# %%
