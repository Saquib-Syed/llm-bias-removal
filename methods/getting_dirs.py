#%%
%cd ../
from transformers import AutoTokenizer, AutoModelForCausalLM
from evals.evals import batched_ave_logit_diff
from dataset.load_dataset import load_data, get_toks, get_tok_pos
from dataset.custom_dataset import PairedInstructionDataset

import json
import torch
from transformer_lens import HookedTransformer
import functools

import transformer_lens.patching as patching
from evals.evals import logit_diff_from_logits
import plotly.express as px

device = "cuda" if torch.cuda.is_available() else "cpu"

#garbage graph, need it for some weird pdf bug in plotly
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()
fig.write_image("random.pdf")

#%%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    default_padding_side="left"
)
model.set_use_attn_result(True)
tokenizer = model.tokenizer
model.eval()
model.to(device)

#%%

with open(f'dataset/gender_bias_ds.json', 'r') as f:
    train_data = json.load(f)

with open(f'dataset/gender_bias_ds_test.json', 'r') as f:
    test_data = json.load(f)

def tokenize_instructions(tokenizer, instructions):
    try:
        return torch.tensor(tokenizer(instructions, padding=True).input_ids)
    except:
        print(instructions)

paired_dataset = PairedInstructionDataset(
    N=60,
    tokenizer=tokenizer,
    tokenize_instructions=tokenize_instructions,
    instruction_templates=train_data['instruction_templates'],
    harmful_substitution_map=train_data['male_sub_map'],
    harmless_substitution_map=train_data['female_sub_map'],
)

train_male_toks = tokenize_instructions(
    tokenizer,
    [
        train_data['instruction_templates'][0].format(occupation=inst)
        for inst in train_data['male_sub_map']["{occupation}"]
    ]
)
train_female_toks = tokenize_instructions(
    tokenizer,
    [
        train_data['instruction_templates'][0].format(occupation=inst)
        for inst in train_data['female_sub_map']['{occupation}']
    ]
)
test_male_toks = tokenize_instructions(
    tokenizer,
    [
        test_data['instruction_templates'][0].format(occupation=inst)
        for inst in test_data['male_sub_map']['{occupation}']
    ]
)
test_female_toks = tokenize_instructions(
    tokenizer,
    [
        test_data['instruction_templates'][0].format(occupation=inst)
        for inst in test_data['female_sub_map']['{occupation}']
    ]
)
he_tok, she_tok, him_tok, his_tok, her_tok = tokenizer.encode(" he she him his her")
#%% Sanity Check
with torch.set_grad_enabled(False):
    male_logit_diff = batched_ave_logit_diff(
            model,
            train_male_toks,
            toks_a=torch.tensor([he_tok]),
            toks_b=torch.tensor([she_tok]),
            batch_size=500,
            do_mean=False
        )
    female_logit_diff = batched_ave_logit_diff(
            model,
            train_female_toks,
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
    "d_model -> batch d_model", batch=train_male_toks.shape[0]
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

_, paired_male_cache = model.run_with_cache(paired_dataset.harmful_dataset.toks)
_, paired_female_cache = model.run_with_cache(paired_dataset.harmless_dataset.toks)

_, male_cache = model.run_with_cache(train_male_toks)
male_final_residual_stream: Float[Tensor, "batch seq d_model"] = male_cache["resid_post", -1]
male_final_token_residual_stream = male_final_residual_stream[:, -1, :]
male_scaled_final_token_residual_stream = male_cache.apply_ln_to_stack(male_final_token_residual_stream, layer=-1, pos_slice=-1)
male_accumulated_residual, _ = male_cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, return_labels=True)
male_per_layer_residual, labels = male_cache.decompose_resid(layer=-1, pos_slice=-1, return_labels=True)

_, female_cache = model.run_with_cache(train_female_toks)
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
        title_x=0.5,
        xaxis_tickangle=-45,
        font=dict(size=12)
    )
)
fig.update_layout(
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    xaxis=dict(title=dict(font=dict(size=20))),
    legend=dict(title=dict(font=dict(size=30))),
    title=dict(font=dict(size=25))
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
        title_x=0.5,
        xaxis_tickangle=-45,
        font=dict(size=12)
    )
)
# update y axis labels, tick labels, legend, and title to have bigger font
fig.update_layout(
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    xaxis=dict(title=dict(font=dict(size=20))),
    legend=dict(title=dict(font=dict(size=30))),
    title=dict(font=dict(size=25))
)
fig.show()
# Save as pdf
fig.write_image("results/per_layer_logit_diffs.pdf")
#%%

# male_logit_diff = male_logit_diff.mean().item()
# female_logit_diff = female_logit_diff.mean().item()
def gender_metric(logits):
    logits = logits[:, -1, :]
    patched = logit_diff_from_logits(logits, toks_a=[he_tok], toks_b=[she_tok])
    return (patched - female_logit_diff) / (male_logit_diff - female_logit_diff)

act_patch_attn_head_out_all_pos = patching.get_act_patch_attn_head_out_all_pos(
    model, 
    paired_dataset.harmless_dataset.toks, 
    paired_male_cache, 
    gender_metric 
)
#%%
import plotly.express as px
fig = px.imshow(
    utils.to_numpy(act_patch_attn_head_out_all_pos), 
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
    labels={"y": "Layer", "x": "Head"}, 
    title="Activation Patching Across All Heads",
    width=600,
    zmin=-0.5,
    zmax=0.5
)
fig.update_layout(
    xaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    yaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    title=dict(font=dict(size=25)),
    coloraxis_colorbar=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    title_x=0.5,
)
fig.update_yaxes(dtick=1)
fig.update_xaxes(dtick=1)
fig.show()
fig.write_image("results/head_act_patch.pdf")
#%%
top_heads = [(10, 9), (9, 7), (4, 3), (11, 8), (9, 2), (6, 0), (8, 11)]
# Patch all the top heads
def top_head_patch_hook(act, hook, cache):
    layer = hook.layer()
    heads_to_patch = [head for (layer_, head) in top_heads if layer_ == layer]
    if len(heads_to_patch) > 0:
        heads_to_patch = torch.tensor(heads_to_patch)
        act[:, :, heads_to_patch, :] = cache[f'blocks.{layer}.attn.hook_z'][:, :, heads_to_patch, :]
    return act

hook_fn = functools.partial(top_head_patch_hook, cache=paired_male_cache)
top_head_patch_logits = model.run_with_hooks(
    paired_dataset.harmless_dataset.toks,
    fwd_hooks=[(f'blocks.{layer}.attn.hook_z', hook_fn) for layer, _ in top_heads]
)
print(gender_metric(top_head_patch_logits))
#%%
# top_head_attn_pattern = torch.stack([
#     paired_male_cache[f'blocks.{layer}.attn.hook_pattern'][:, head, -1, :].mean(dim=0)
#     for layer, head in top_heads
# ])
# pttn_labels = ['(0) The', '(1) {OCCUPATION}', '(2) {OCCUPATION}', '(3) said', '(4) that']

# imshow(
#     top_head_attn_pattern, 
#     labels={"x": "Position", "y": "Head"},
#     x=pttn_labels,
#     y=[f"L{layer}H{head}" for layer, head in top_heads],
#     title="Top Heads Attention Pattern",
#     width=700,
#     zmin=0,
#     zmax=1,
#     color_continuous_scale='Blues',
# )

#%% PCA of activations of the most important components
from sklearn.decomposition import PCA
l10h9_acts = torch.cat(
    [
        male_cache["blocks.10.attn.hook_result"][:, -1, 9, :],
        female_cache["blocks.10.attn.hook_result"][:, -1, 9, :]
    ],
    dim=0
)

pca = PCA(n_components=2)
pca_acts = pca.fit_transform(utils.to_numpy(l10h9_acts))
male_cache_len = male_cache['blocks.0.attn.hook_result'].shape[0]
# Plot the PCA
fig = go.Figure(
    data=[
        go.Scatter(
            x=pca_acts[:male_cache_len, 0],
            y=pca_acts[:male_cache_len, 1],
            mode="markers",
            name="Male Activations"
        ),
        go.Scatter(
            x=pca_acts[male_cache_len:, 0],
            y=pca_acts[male_cache_len:, 1],
            mode="markers",
            name="Female Activations"
        ),
    ],
    layout=go.Layout(
        title="PCA of Activations for Layer 10 Head 9",
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        hovermode="closest",
        # xaxis_range=[-1.5, 1.5],
        # yaxis_range=[-1.5, 1.5],
        title_x=0.5,
        legend=dict(font=dict(size=20)),
    )
)
fig.update_layout(
    yaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    xaxis=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    title=dict(font=dict(size=25)),
)
fig.show()
fig.write_image("results/pca_l10h9.pdf")
# Perform PCA on the acts, which are shape (batch, d_head)
pca_components = {}
for (layer, head) in top_heads:
    stacked_head_act = torch.cat([
        male_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :], 
        female_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :]
    ], dim=0).cpu()
    pca = PCA(n_components=1)
    pca_acts = pca.fit_transform(utils.to_numpy(stacked_head_act))
    pca_components[(layer, head)] = pca.components_[0]


#%% Activation patching on residual stream
resid_pre_act_patch_results = patching.get_act_patch_resid_pre(
    model,
    paired_dataset.harmless_dataset.toks, 
    paired_male_cache, 
    gender_metric 
)
labels = [f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(paired_dataset.harmless_dataset.toks[0]))]

#%%

pttn_labels = ['(0) The', '(1) {OCCUPATION}', '(2) {OCCUPATION}', '(3) said', '(4) that']
fig = px.imshow(
    utils.to_numpy(resid_pre_act_patch_results), 
    color_continuous_scale="Blues",
    labels={"x": "Position", "y": "Layer"},
    x=pttn_labels,
    title="Activation Patching on the Residual Stream",
    width=600,
    height=500
)
fig.update_layout(
    xaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    yaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    title=dict(font=dict(size=25)),
    coloraxis_colorbar=dict(title=dict(font=dict(size=20)), tickfont=dict(size=20)),
    title_x=0.5,
)
fig.update_yaxes(dtick=1)
fig.show()
fig.write_image("results/resid_pre_act_patch.pdf")
#%% Act diffs
# male - female directions
act_diffs = {
    (layer, head): male_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :] - female_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :]
    for layer, head in top_heads
}
#%% Probing
from sklearn.linear_model import LogisticRegression
# We train 7 probes on the stacked activations of each head
probe_dirs = {}

for (layer, head) in top_heads:
    male_batch_len = male_cache[f'blocks.{layer}.attn.hook_result'].shape[0]
    female_batch_len = female_cache[f'blocks.{layer}.attn.hook_result'].shape[0]

    stacked_head_act = torch.cat([
        male_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :], 
        female_cache[f'blocks.{layer}.attn.hook_result'][:, -1, head, :]
    ], dim=0).cpu()
    clf = LogisticRegression(max_iter=1000)
    clf.fit(stacked_head_act, torch.tensor([0] * male_batch_len + [1] * female_batch_len))
    probe_dirs[(layer, head)] = clf.coef_

#%% Ablations
import numpy as np
# We take all these directions and ablate them from the heads
# Then, we measure the logit diff across a bunch of examples

test_male_logit_diff = batched_ave_logit_diff(
    model,
    test_male_toks,
    toks_a=torch.tensor([he_tok]),
    toks_b=torch.tensor([she_tok]),
    batch_size=500,
)
test_female_logit_diff = batched_ave_logit_diff(
    model,
    test_female_toks,
    toks_a=torch.tensor([he_tok]),
    toks_b=torch.tensor([she_tok]),
    batch_size=500,
)
def ablating_male_metric(logits):
    logits = logits[:, -1, :]
    patched = logit_diff_from_logits(logits, toks_a=[he_tok], toks_b=[she_tok])
    return (patched - test_male_logit_diff) / (test_female_logit_diff - test_male_logit_diff)

def ablating_female_metric(logits):
    logits = logits[:, -1, :]
    patched = logit_diff_from_logits(logits, toks_a=[he_tok], toks_b=[she_tok])
    return (patched - test_female_logit_diff) / (test_male_logit_diff - test_female_logit_diff)

def ablate_dir_hook(act, hook, dirs, reverse_dir=False):
    layer = hook.layer()
    for (layer_, head), d in dirs.items():
        try:
            d = d.mean(dim=0)
            d /= d.norm()
        except:
            d = torch.tensor(d, dtype=torch.float32).to(device)
            d = d.mean(dim=0)
            d /= d.norm()
        if reverse_dir:
            d = -d
        if layer_ == layer:
            print(layer, head)
            act[:, -1, head, :] = act[:, -1, head, :] - ((act[:, -1, head, :] @ d).unsqueeze(-1) * d)
    return act

act_diff_logits = model.run_with_hooks(
    test_male_toks,
    fwd_hooks=[
        (
            f'blocks.{layer}.attn.hook_result', 
            functools.partial(
                ablate_dir_hook, 
                dirs=probe_dirs,
                reverse_dir=False
            )
        )
        for layer, _ in top_heads
    ]
)
print(logit_diff_from_logits(act_diff_logits[:, -1, :], toks_a=[he_tok], toks_b=[she_tok]))

# %% Plan B: mean ablate the important heads
import numpy as np
rand_heads = [
    (5, 4), (6, 9), (7, 6), (8, 3), (9, 5), (10, 10), (11, 1)
]
def head_ablation_hook(act, hook, male_cache, female_cache, mult=1):
    layer = hook.layer()
    heads_to_ablate = [head for (layer_, head) in rand_heads if layer_ == layer]
    if len(heads_to_ablate) > 0:
        heads_to_ablate = torch.tensor(heads_to_ablate)
        mean_act = torch.cat([
            male_cache[f'blocks.{layer}.attn.hook_z'],
            mult * female_cache[f'blocks.{layer}.attn.hook_z'],
        ], dim=0).mean(0).unsqueeze(0)
        act[:, -1, heads_to_ablate, :] = mean_act[:, -1, heads_to_ablate, :]
    return act


for mult in np.linspace(-20, 5, 11):
    hook_fn = functools.partial(
        head_ablation_hook,
        male_cache=paired_male_cache,
        female_cache=paired_female_cache,
        mult=mult
    )

    ablate_logits = model.run_with_hooks(
        test_female_toks,
        fwd_hooks=[
            (
                f'blocks.{layer}.attn.hook_z', 
                hook_fn
            )
            for layer, _ in top_heads
        ]
    )
    print(mult, logit_diff_from_logits(ablate_logits[:, -1, :], toks_a=[he_tok], toks_b=[she_tok]))

#%%
female_mult=1.6
hook_fn = functools.partial(
    head_ablation_hook,
    male_cache=paired_male_cache,
    female_cache=paired_female_cache,
    mult=female_mult
)

ablate_logits = model.run_with_hooks(
    test_female_toks,
    fwd_hooks=[
        (
            f'blocks.{layer}.attn.hook_z', 
            hook_fn
        )
        for layer, _ in top_heads
    ]
)
female_ablate_logit_diff = logit_diff_from_logits(
    ablate_logits[:, -1, :], 
    toks_a=[he_tok], 
    toks_b=[she_tok],
    do_mean=False
)
female_ablate_logit_diff = [l.item() for l in female_ablate_logit_diff]

male_mult=2.4
hook_fn = functools.partial(
    head_ablation_hook,
    male_cache=paired_male_cache,
    female_cache=paired_female_cache,
    mult=male_mult
)

ablate_logits = model.run_with_hooks(
    test_male_toks,
    fwd_hooks=[
        (
            f'blocks.{layer}.attn.hook_z', 
            hook_fn
        )
        for layer, _ in top_heads
    ]
)
male_ablate_logit_diff = logit_diff_from_logits(
    ablate_logits[:, -1, :], 
    toks_a=[he_tok], 
    toks_b=[she_tok],
    do_mean=False
)
male_ablate_logit_diff = [l.item() for l in male_ablate_logit_diff]
with open('results/rand_ablate.json', 'w') as f:
    json.dump(
        {'male_jobs': male_ablate_logit_diff, 
         'female_jobs': female_ablate_logit_diff}, f)
# %%
