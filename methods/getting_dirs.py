#%%
%cd ../
from transformers import AutoTokenizer, AutoModelForCausalLM
from evals.evals import batched_ave_logit_diff
from dataset.load_dataset import load_data, get_toks, get_tok_pos

import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
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

he_tok, she_tok, him_tok, his_tok, her_tok = tokenizer.encode(" he she him his her")
#%% Sanity Check

N = 500
with torch.set_grad_enabled(False):
    male_logit_diff = batched_ave_logit_diff(
            model,
            dataset.harmful_dataset.toks[:N],
            toks_a=torch.tensor([he_tok]),
            toks_b=torch.tensor([she_tok]),
            batch_size=500,
            do_mean=False
        )
    female_logit_diff = batched_ave_logit_diff(
            model,
            dataset.harmless_dataset.toks[:N],
            toks_a=torch.tensor([he_tok]),
            toks_b=torch.tensor([she_tok]),
            batch_size=500,
            do_mean=False
        )

    male_logit_diff = male_logit_diff.mean().item()
    female_logit_diff = female_logit_diff.mean().item()
    print(male_logit_diff, female_logit_diff)
#%% Logit diff across model components
from plotly_utils import imshow, line, scatter, bar
import plotly.graph_objects as go
from jaxtyping import Float
from torch import Tensor
import einops 
from transformer_lens import utils

logit_diff_dir = einops.repeat(
    model.tokens_to_residual_directions(he_tok) - model.tokens_to_residual_directions(she_tok),
    "d_model -> batch d_model", batch=N
)
def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"], 
    cache,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_dir,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given 
    stack of components in the residual stream.
    '''

    scaled_residual_stream = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    average_logit_diff = einops.einsum(
        scaled_residual_stream, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / residual_stack.size(-2)

    return average_logit_diff

_, male_cache = model.run_with_cache(dataset.harmful_dataset.toks[:N])
male_final_residual_stream: Float[Tensor, "batch seq d_model"] = male_cache["resid_post", -1]
male_final_token_residual_stream = male_final_residual_stream[:, -1, :]
male_scaled_final_token_residual_stream = male_cache.apply_ln_to_stack(male_final_token_residual_stream, layer=-1, pos_slice=-1)
male_accumulated_residual, _ = male_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
male_per_layer_residual, labels = male_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)

_, female_cache = model.run_with_cache(dataset.harmless_dataset.toks[:N])
female_final_residual_stream: Float[Tensor, "batch seq d_model"] = female_cache["resid_post", -1]
female_final_token_residual_stream = female_final_residual_stream[:, -1, :]
female_scaled_final_token_residual_stream = female_cache.apply_ln_to_stack(female_final_token_residual_stream, layer=-1, pos_slice=-1)
female_accumulated_residual, acc_labels = female_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
female_per_layer_residual, dec_labels = female_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)

# Plot logit diffs across resid stream
male_logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
    male_accumulated_residual, 
    male_cache,
)
male_per_layer_logit_diffs = residual_stack_to_logit_diff(
    male_per_layer_residual, 
    male_cache,
)

female_logit_lens_logit_diffs: Float[Tensor, "component"] = residual_stack_to_logit_diff(
    female_accumulated_residual, 
    female_cache,
)
female_per_layer_logit_diffs = residual_stack_to_logit_diff(
    female_per_layer_residual, 
    female_cache,
)

#%%
# Plot logit diffs from each layer
fig = go.Figure(
    data=[
        go.Scatter(
            x=acc_labels,
            y=utils.to_numpy(male_logit_lens_logit_diffs),
            mode="lines",
            name="Male"
        ),
        go.Scatter(
            x=acc_labels,
            y=utils.to_numpy(female_logit_lens_logit_diffs),
            mode="lines",
            name="Female"
        )
    ],
    layout=go.Layout(
        title="Accumulated Logit Difference From Each Layer",
        xaxis_title="Layer",
        yaxis_title="Logit Diff (Male - Female)",
        hovermode="x unified",
    )
)
fig.show()
# Save as pdf
fig.write_image("results/accumulated_logit_diffs.pdf")

fig = go.Figure(
    data=[
        go.Scatter(
            x=dec_labels,
            y=utils.to_numpy(male_per_layer_logit_diffs),
            mode="lines",
            name="Male"
        ),
        go.Scatter(
            x=dec_labels,
            y=utils.to_numpy(female_per_layer_logit_diffs),
            mode="lines",
            name="Female"
        )
    ],
    layout=go.Layout(
        title="Logit Difference Contributions Per Layer",
        xaxis_title="Layer",
        yaxis_title="Logit Diff (Male - Female)",
        hovermode="x unified",
    )
)
fig.show()
# Save as pdf
fig.write_image("results/per_layer_logit_diffs.pdf")
#%%
import transformer_lens.patching as patching
from evals.evals import logit_diff_from_logits

# male_logit_diff = male_logit_diff.mean().item()
# female_logit_diff = female_logit_diff.mean().item()
def gender_metric(logits):
    logits = logits[:, -1, :]
    patched = logit_diff_from_logits(logits, toks_a=[he_tok], toks_b=[she_tok])
    return (patched - female_logit_diff) / (male_logit_diff - female_logit_diff)

act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    dataset.harmless_dataset.toks[:N], 
    male_cache, 
    gender_metric 
)
#%%
imshow(
    act_patch_attn_head_out_all_pos, 
    labels={"y": "Layer", "x": "Head"}, 
    title="attn_head_out Activation Patching (All Pos)",
    width=600,
    margin=dict(l=10, r=20, t=50, b=20)
)

#%% PCA of activations of the most important components
l10h9_acts = torch.cat(
    [
        male_cache["blocks.10.attn.hook_z"][:, -1, 9, :],
        female_cache["blocks.10.attn.hook_z"][:, -1, 9, :]
    ],
    dim=0
)

# Perform PCA on the acts, which are shape (batch, d_head)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca_acts = pca.fit_transform(utils.to_numpy(l10h9_acts))

# Plot the PCA
fig = go.Figure(
    data=[
        go.Scatter(
            x=pca_acts[:N, 0],
            y=pca_acts[:N, 1],
            mode="markers",
            name="Male Activations"
        ),
        go.Scatter(
            x=pca_acts[N:, 0],
            y=pca_acts[N:, 1],
            mode="markers",
            name="Female Activations"
        ),
    ],
    layout=go.Layout(
        title="PCA of Activations of the Most Important Components",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        hovermode="closest",
    )
)
fig.show()
#%% Act diffs

#%% Probing

#%% Ablations

#%% Orthogonalization
