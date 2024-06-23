#%%

import pandas as pd
#%%
# hf url: https://huggingface.co/datasets/flax-sentence-embeddings/Gender_Bias_Evaluation_Set
# paper: arxiv:1906.00591

df = pd.read_csv("hf://datasets/flax-sentence-embeddings/Gender_Bias_Evaluation_Set/bias_evaluation.csv")

# %%

def load_data():
    df = pd.read_csv(
        "hf://datasets/flax-sentence-embeddings/Gender_Bias_Evaluation_Set/bias_evaluation.csv"
    )
    return df

# Relevant token positions
# {pronoun}: "he" or "she" or "his" or "her"
# {occupation}: the job in the sentence

from collections import defaultdict

def get_toks(df, tokenizer):
    toks = tokenizer(
        [
            sentence for sentence in df['base_sentence']
        ],
        padding = True
    ).input_ids
    occ_toks = [
        tokenizer.encode(" " + occupation)[1:] for occupation in df['occupation']
    ]
    return toks, occ_toks

def get_tok_pos(toks, occ_toks, tokenizer):
    def find_sub_list(sl,l):
        if type(sl) == int:
            sl = [sl]

        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                return (ind,ind+sll-1)

        return (None, None)

    tok_pos = defaultdict(list) # tok_pos: [start, end]

    _, he_tok, she_tok, him_tok, his_tok, her_tok = tokenizer.encode(" he she him his her")

    for occ_tok, sentence_tok in zip(occ_toks, toks):
        # Finding occupation in the sentence
        occ_pos, occ_len = find_sub_list(occ_tok, sentence_tok)
        tok_pos["{occupation}"].append((occ_pos, occ_len))

        found = False
        for pronoun_tok in [he_tok, she_tok, him_tok, his_tok, her_tok]:
            # Finding pronoun in the sentence
            pronoun_pos, pronoun_len = find_sub_list(pronoun_tok, sentence_tok)
            if pronoun_pos is not None:
                tok_pos["{pronoun}"].append((pronoun_pos, pronoun_len)) 
                found = True
                break
        if not found:
            print(
                f"Pronoun not found in sentence\n{[f'{t}:{tokenizer.decode(t)}' for t in sentence_tok]}")

    return tok_pos

# def get_paired_toks_with_pos(toks, tok_pos, tokenizer):
#     # We pair sentences with the same length occupation and pronoun
#     # Return (sentence_a, tok_pos_a, sentence_b, tok_pos_b) lists

#     # For each tokenized sentence, find another sentence with the same length occupation and pronoun
# %%
