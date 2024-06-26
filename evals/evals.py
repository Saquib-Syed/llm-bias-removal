import torch
from tqdm.auto import tqdm

# Logit diff
def ave_logit_diff(
    model, 
    toks, 
    toks_a, 
    toks_b,
    do_mean=True
):
    logits = model(toks)
    logits = logits[:, -1, :]

    if do_mean:
        return (logits[:, toks_a].mean(dim=-1) - logits[:, toks_b].mean(dim=-1)).mean()
    return list(logits[:, toks_a].mean(dim=-1) - logits[:, toks_b].mean(dim=-1))

def logit_diff_from_logits(logits, toks_a, toks_b, do_mean=True):
    if do_mean:
        return (logits[:, toks_a].mean(dim=-1) - logits[:, toks_b].mean(dim=-1)).mean()
    return list(logits[:, toks_a].mean(dim=-1) - logits[:, toks_b].mean(dim=-1))

def batched_ave_logit_diff(
    model, 
    toks, 
    toks_a, 
    toks_b, 
    batch_size,
    do_mean=True
):
    logit_diff = []
    for i in tqdm(range(0, len(toks), batch_size)):
        toks_slice = toks[i:i+batch_size]
        if do_mean:
            logit_diff.append(ave_logit_diff(
                model, 
                toks_slice, 
                toks_a, 
                toks_b,
                do_mean
            ))
        else:
            logit_diff.extend(ave_logit_diff(
                model, 
                toks_slice, 
                toks_a, 
                toks_b,
                do_mean
            ))

    if do_mean:
        return torch.tensor(logit_diff).mean()
    else:
        return torch.tensor(logit_diff)

def logit_diff_on_gender(
    df, 
    model, 
    toks, 
    toks_a, 
    toks_b, 
    gender="male", 
    batch_size=None
):
    gender_idx = torch.tensor(df.index[df['stereotypical_gender'] == gender])

    if batch_size is None:
        return ave_logit_diff(model, toks[gender_idx], toks_a, toks_b)
    return batched_ave_logit_diff(model, toks[gender_idx], toks_a, toks_b, batch_size)
