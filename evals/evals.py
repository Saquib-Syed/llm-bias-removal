import torch
from tqdm.auto import tqdm

# Logit diff
def ave_logit_diff(
    model, 
    toks, 
    pos, 
    toks_a, 
    toks_b
):
    logits = model(toks)
    logits = logits[range(logits.shape[0]), pos, :]

    return (logits[:, toks_a] - logits[:, toks_b]).mean()

def batched_ave_logit_diff(
    model, 
    toks, 
    pos, 
    toks_a, 
    toks_b, 
    batch_size
):
    logit_diff = 0
    for i in tqdm(range(0, len(toks), batch_size)):
        logit_diff += ave_logit_diff(
            model, 
            toks[i:i+batch_size], 
            pos[i:i+batch_size],
            toks_a, 
            toks_b
        )

    return logit_diff / (len(toks) // batch_size)

def logit_diff_on_gender(
    df, 
    model, 
    toks, 
    pos,
    toks_a, 
    toks_b, 
    gender="male", 
    batch_size=None
):
    gender_idx = torch.tensor(df.index[df['stereotypical_gender'] == gender])

    if batch_size is None:
        return ave_logit_diff(model, toks[gender_idx], pos[gender_idx], toks_a, toks_b)
    return batched_ave_logit_diff(model, toks[gender_idx], pos[gender_idx], toks_a, toks_b, batch_size)
