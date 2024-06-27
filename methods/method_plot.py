#%%
%cd ~/llm-bias-removal
import plotly.graph_objects as go
import json

import plotly.express as px


#garbage graph, need it for some weird pdf bug in plotly
fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
fig.show()
fig.write_image("random.pdf")

with open(f"results/baseline.json", "r") as f:
    baseline = json.load(f)

with open(f"results/mean_ablate.json", "r") as f:
    mean_ablate = json.load(f)

with open(f"results/rand_ablate.json", "r") as f:
    rand_ablate = json.load(f)
#%%
import numpy as np
baseline_male = np.array(baseline['male_jobs'])
baseline_female = np.array(baseline['female_jobs'])
mean_ablate_male = np.array(mean_ablate['male_jobs'])
mean_ablate_female = np.array(mean_ablate['female_jobs'])
rand_ablate_male = np.array(rand_ablate['male_jobs'])
rand_ablate_female = np.array(rand_ablate['female_jobs'])

fig = go.Figure()
fig.add_trace(go.Bar(
    name='Male Jobs',
    x=['Baseline', 'Top Heads <br> Mean Ablated', 'Random Heads <br> Mean Ablated'], 
    y=[baseline_male.mean(), mean_ablate_male.mean(), rand_ablate_male.mean()],
    error_y=dict(type='data', array=[baseline_male.std(), mean_ablate_male.std(), rand_ablate_male.std()])
))
fig.add_trace(go.Bar(
    name='Female Jobs',
    x=['Baseline', 'Top Heads <br> Mean Ablated', 'Random Heads <br> Mean Ablated'], 
    y=[baseline_female.mean(), mean_ablate_female.mean(), rand_ablate_female.mean()],
    error_y=dict(type='data', array=[baseline_female.std(), mean_ablate_female.std(), rand_ablate_female.std()])
))
fig.update_layout(
    barmode='group',
    title='Logit Difference in Predicted Gender',
    xaxis_title='Ablation Method',
    yaxis_title='Logit Difference',
)
fig.update_layout(
    xaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    yaxis=dict(title=dict(font=dict(size=25)), tickfont=dict(size=15)),
    title=dict(font=dict(size=25)),
    legend=dict(font=dict(size=15)),
    title_x=0.5,
)
fig.show()
fig.write_image("results/ablations.pdf")
# %%
