# KEXS — Knowledge-Enhanced Explainable Machine Learning for Predicting Startup Success

Code, data and results for the short communication
**"Knowledge-Enhanced Explainable Machine Learning for Predicting Startup Success"**
(S.K. Bhagam, submitted to *Knowledge-Based Systems*).

KEXS injects five layers of domain knowledge into gradient-boosted trees to predict whether a
venture-backed startup is acquired (success) or closes (failure):

| Layer | Knowledge | Where |
|---|---|---|
| K1 | Expert sector ontology (800+ categories → 10 macro-sectors, capital intensity) | `kexs.k1_ontology` |
| K2 | Category knowledge graph → PPMI → 16-d spectral embedding | `kexs.k2_category_graph` |
| K3 | Venture-staging rules (stage reached, investor mix, round velocity, …) | `kexs.k3_financing` |
| K4 | Entrepreneurial-ecosystem statistics (density, capital, location quotient) | `kexs.k4_ecosystem` |
| K5 | Monotone constraints encoding directional expert priors | `kexs.MONOTONE_UP`, `kexs.make_model` |

Explanations: exact **group-Shapley** attribution over feature groups (`kexs.group_shapley`),
partial dependence, and an individual-level monotonicity check.

## Repository layout

```
kexs.py        # the method: data prep, K1–K5, models, group Shapley
run.py         # reproduces every number, table and figure of the paper
data/          # public Crunchbase export (Kaggle redistribution)
results/       # generated tables (csv), figures (png 600 dpi + pdf), metrics.json
paper/         # manuscript.docx, highlights.docx
```

## Reproduce

```bash
pip install -r requirements.txt
python run.py          # ~25 min on 2 CPU cores, fully deterministic (seed 42)
python run.py --tune   # re-runs the GBDT hyper-parameter selection and ontology-coverage check
```

## Main results (from `results/table2_performance.csv`)

| Model | CV ROC-AUC | Temporal AUC (2010–2014) |
|---|---|---|
| GBDT, raw features | 0.800 ± 0.014 | 0.771 |
| RF, raw features | 0.798 ± 0.012 | 0.767 |
| **KEXS** | **0.812 ± 0.013** | **0.783** |

KEXS improves the identical data-only model by +0.012 AUC (corrected resampled t-test p < 0.001;
temporal 95% CI [+0.003, +0.022]) and removes all individual-level violations of the encoded
expert priors (25–60% of firms in the unconstrained model).

## Data

`data/investments_VC.csv` is the public *StartUp Investments (Crunchbase)* dataset from Kaggle
(https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase), a Crunchbase export
up to 2014. SHA-256: `4232d61be9e91e99194cb93b01688448295a0d38ce3f0496756eb70c81b9e9b6`.
Check the dataset's licence terms before reuse; if in doubt, download it from Kaggle into `data/`.

## Licence

Code: MIT (see `LICENSE`).
