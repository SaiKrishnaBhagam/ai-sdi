"""Reproduce every number, table and figure in the paper.

    python run.py            # full run (~25 min on 2 CPU cores)
    python run.py --tune     # re-run the GBDT hyper-parameter selection (Base features only)
                             # and the ontology-coverage check reported in Section 2

Outputs go to results/: table1_knowledge_layers.csv, table2_performance.csv,
metrics.json and figure1..figure4 (.png at 600 dpi and vector .pdf).
"""
from __future__ import annotations

import json
import os
import sys

if "--tune" in sys.argv:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import kexs
    X, y, G, _, d_all = kexs.build("data/investments_VC.csv")
    cats = d_all["cats"].explode().dropna().value_counts()
    cov = sum(n for c, n in cats.items() if kexs.sector_of(c)) / cats.sum()
    print(f"K1 ontology: {len(cats)} categories, {100 * cov:.1f}% of category mentions mapped")
    cv = StratifiedKFold(5, shuffle=True, random_state=7)          # split differs from evaluation
    grid = [(0.05, 400, 31, 30, 1), (0.03, 300, 15, 50, 3), (0.02, 500, 15, 50, 3),
            (0.03, 300, 7, 80, 5), (0.05, 150, 15, 40, 1), (0.02, 800, 7, 50, 3)]
    for lr, it, leaves, msl, l2 in grid:
        m = HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                           min_samples_leaf=msl, l2_regularization=l2,
                                           early_stopping=False, random_state=kexs.SEED)
        auc = cross_val_score(m, X[G["Base"]], y, cv=cv, scoring="roc_auc").mean()
        print(f"lr={lr} trees={it} leaves={leaves} min_leaf={msl} l2={l2}:  Base AUC {auc:.4f}")
    sys.exit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

import kexs

DATA = "data/investments_VC.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(kexs.SEED)

# palette (validated categorical slots 1-2) + recessive ink
C_KE, C_BASE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for paired CV scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1) * (1 / k + n_test / n_train)
    t = d.mean() / np.sqrt(var)
    return d.mean(), t, 2 * stats.t.sf(abs(t), k - 1)


# ---------------------------------------------------------------------------
X, y, G, meta, d_all = kexs.build(DATA)
ALL = [c for cols in G.values() for c in cols]
n = len(y)
print(f"labelled firms: {n}  (acquired={int(y.sum())}, closed={int((1 - y).sum())}); "
      f"unlabelled reference firms: {len(d_all) - n}")

# ---------------- Table 1: knowledge layers --------------------------------
t1 = pd.DataFrame([
    ["Base", "Raw Crunchbase fields: log amounts per funding instrument, rounds, founding and first-funding year, top-30 market and top-15 country dummies", len(G["Base"]), "Data only (no knowledge)"],
    ["K1", "Expert sector ontology: 800+ categories mapped to 10 macro-sectors, sector breadth, capital-intensity level", len(G["K1 ontology"]), "Industry taxonomy; capital-intensity view of venture finance"],
    ["K2", "Category knowledge graph: co-occurrence graph of categories, PPMI weighting, 16-d spectral embedding averaged per firm", len(G["K2 category graph"]), "Market relatedness / knowledge-graph embedding"],
    ["K3", "Venture-staging knowledge: highest stage reached, institutional / informal / non-dilutive capital shares, instrument diversity, round velocity, time to first funding", len(G["K3 financing"]), "Staged financing and investor-signalling theory"],
    ["K4", "Ecosystem knowledge: regional and national start-up density, regional capital, market crowding, hub membership, sector location quotient", len(G["K4 ecosystem"]), "Agglomeration / entrepreneurial-ecosystem theory"],
    ["K5", "Monotone constraints on stage, institutional share, total funding and rounds (non-decreasing effect on exit odds)", 4, "Prior domain knowledge as model constraint"],
], columns=["Layer", "Content", "Features", "Knowledge source"])
t1.to_csv(f"{OUT}/table1_knowledge_layers.csv", index=False)

# ---------------- Experiment 1: repeated CV benchmark ----------------------
rskf = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=kexs.SEED).split(X, y))
configs = [(m, fs, False) for m in ["LR", "MLP", "RF", "GBDT"] for fs in ["Base", "Base+K"]]
configs.append(("GBDT", "Base+K", True))                  # KEXS = GBDT + K1-K4 + K5
cols_of = {"Base": G["Base"], "Base+K": ALL}
fold_scores, oof_rep1 = {}, {}
for m, fs, mono in configs:
    key = "KEXS (GBDT+K+mono)" if mono else f"{m} | {fs}"
    sc = {"auc": [], "ap": [], "brier": [], "bacc": []}
    oof = np.zeros(n)
    for i, (tr, te) in enumerate(rskf):
        cols = cols_of[fs]
        mdl = kexs.make_model(m, cols, mono).fit(X.iloc[tr][cols], y.iloc[tr])
        p = mdl.predict_proba(X.iloc[te][cols])[:, 1]
        sc["auc"].append(roc_auc_score(y.iloc[te], p))
        sc["ap"].append(average_precision_score(y.iloc[te], p))
        sc["brier"].append(brier_score_loss(y.iloc[te], p))
        sc["bacc"].append(balanced_accuracy_score(y.iloc[te], p >= 0.5))
        if i < 5:
            oof[te] = p
    fold_scores[key], oof_rep1[key] = sc, oof
    print(f"{key:24s} AUC {np.mean(sc['auc']):.4f} ± {np.std(sc['auc']):.4f}")

n_tr, n_te = len(rskf[0][0]), len(rskf[0][1])

# ---------------- Experiment 2: strict temporal hold-out -------------------
# Knowledge structures rebuilt from firms first funded <= 2009 only.
Xt, yt, Gt, mt, _ = kexs.build(DATA, ref_max_year=2009)
ALLt = [c for cols in Gt.values() for c in cols]
yr = mt["first_funding_at"].dt.year
tr_t, te_t = np.where(yr <= 2009)[0], np.where(yr >= 2010)[0]
temporal = {}
for m, fs, mono in configs:
    key = "KEXS (GBDT+K+mono)" if mono else f"{m} | {fs}"
    cols = Gt["Base"] if fs == "Base" else ALLt
    p = kexs.make_model(m, cols, mono).fit(Xt.iloc[tr_t][cols], yt.iloc[tr_t]).predict_proba(Xt.iloc[te_t][cols])[:, 1]
    temporal[key] = p
yte = yt.iloc[te_t].values


def boot_auc_diff(pa, pb, B=2000):
    idx = rng.integers(0, len(yte), size=(B, len(yte)))
    diffs = np.array([roc_auc_score(yte[i], pa[i]) - roc_auc_score(yte[i], pb[i]) for i in idx])
    return np.percentile(diffs, [2.5, 97.5]), np.mean(diffs <= 0)


# ---------------- Table 2: performance -------------------------------------
rows = []
for key, sc in fold_scores.items():
    m = key.split(" |")[0] if "|" in key else "GBDT"
    base_key = f"{m} | Base"
    row = {"Model": key,
           "ROC-AUC": f"{np.mean(sc['auc']):.3f} ± {np.std(sc['auc']):.3f}",
           "PR-AUC": f"{np.mean(sc['ap']):.3f} ± {np.std(sc['ap']):.3f}",
           "Brier": f"{np.mean(sc['brier']):.3f}",
           "Bal. acc.": f"{np.mean(sc['bacc']):.3f}",
           "Temporal AUC": f"{roc_auc_score(yte, temporal[key]):.3f}"}
    if key != base_key:
        dm, t, pval = corrected_ttest(sc["auc"], fold_scores[base_key]["auc"], n_tr, n_te)
        ci, p_boot = boot_auc_diff(temporal[key], temporal[base_key])
        row.update({"ΔAUC vs Base (CV)": f"{dm:+.3f}", "p (corr. t)": f"{pval:.3g}",
                    "ΔAUC temporal [95% CI]": f"{roc_auc_score(yte, temporal[key]) - roc_auc_score(yte, temporal[base_key]):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"})
    rows.append(row)
t2 = pd.DataFrame(rows).fillna("—")
t2.to_csv(f"{OUT}/table2_performance.csv", index=False)
print(t2.to_string(index=False))

# ---------------- Experiment 3: group Shapley + leave-one-group-out --------
skf = list(StratifiedKFold(5, shuffle=True, random_state=kexs.SEED).split(X, y))
phi, cache = kexs.group_shapley(X, y, G, skf, "GBDT")
v_all = cache[frozenset(G)]
lofo = {g: v_all - cache[frozenset(set(G) - {g})] for g in G}
print("group Shapley:", {k: round(v, 4) for k, v in phi.items()}, "v(all)=", round(v_all, 4))

# ---------------- Experiment 4: partial dependence of KEXS ------------------
kexs_model = kexs.make_model("GBDT", ALL, True).fit(X[ALL], y)
free_model = kexs.make_model("GBDT", ALL, False).fit(X[ALL], y)     # same, without K5
pd_feats = [("k3_max_stage", "K3: highest stage reached\n(0 none, 1 seed/angel, 2 A … 5 D+)"),
            ("k3_time_to_first_funding", "K3: years from founding\nto first funding"),
            ("k4_log_region_density", "K4: log start-up density\nof home region")]
def partial_dep(model, Xf, col, grid):
    """Average predicted log-odds when `col` is set to each grid value (Friedman, 2001)."""
    out = []
    for g in grid:
        Z = Xf.copy(); Z[col] = g
        p = np.clip(model.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        out.append(float(np.mean(np.log(p / (1 - p)))))
    return out


def ice_violations(model, Xf, col, n_grid=12):
    """Share of firms whose individual prediction DEcreases somewhere along `col`."""
    vals = Xf[col].dropna()
    grid = np.sort(vals.unique()) if vals.nunique() <= 8 else np.linspace(*np.percentile(vals, [2, 98]), n_grid)
    P = np.column_stack([model.predict_proba(Xf.assign(**{col: g}))[:, 1] for g in grid])
    return float(np.mean((np.diff(P, axis=1) < -1e-9).any(axis=1)))


ice = {f: {"no_K5": ice_violations(free_model, X[ALL], f), "KEXS": ice_violations(kexs_model, X[ALL], f)}
       for f in kexs.MONOTONE_UP}
print("individual-level monotonicity violations:", ice)

pdp, pdp_free = {}, {}
for f, _ in pd_feats:
    vals = X[f].dropna()
    grid = np.sort(vals.unique()) if vals.nunique() <= 8 else np.linspace(*np.percentile(vals, [2, 98]), 25)
    pdp[f] = (grid.tolist(), partial_dep(kexs_model, X[ALL], f, grid))
    pdp_free[f] = (grid.tolist(), partial_dep(free_model, X[ALL], f, grid))

# ---------------- Figures --------------------------------------------------
# Figure 1: framework
fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.2)
def box(x, y0, w, h, txt, fc, ec, bold=False):
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=6.8, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)
def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
box(0.05, 2.85, 2.2, 1.05, f"Crunchbase export\n{len(d_all):,} firms\n(labelled + unresolved)", "#f2f1ee", MUTED)
box(0.05, 0.35, 2.2, 1.05, "Expert knowledge:\nsector ontology,\nVC staging, ecosystems", "#f2f1ee", MUTED)
ks = [("K1  Sector ontology", 3.35), ("K2  Category graph\n(PPMI + spectral embedding)", 2.45),
      ("K3  Venture-staging rules", 1.55), ("K4  Ecosystem statistics", 0.65)]
for t, yy in ks:
    box(3.0, yy - 0.05, 2.6, 0.72, t, "#e8f1fb", C_KE)
for _, yy in ks:
    arrow(2.25, 3.37, 3.0, yy + 0.31); arrow(2.25, 0.87, 3.0, yy + 0.31)
box(6.25, 1.5, 1.95, 1.4, "Gradient-boosted\ntrees with\nK5 monotone\nconstraints", "#e8f1fb", C_KE, True)
for _, yy in ks:
    arrow(5.6, yy + 0.31, 6.25, 2.2)
box(8.75, 2.55, 1.6, 1.0, "Exit probability\nP(acquired)", "#f2f1ee", MUTED)
box(8.75, 0.75, 1.6, 1.3, "Explanations:\ngroup Shapley,\npartial\ndependence", "#f2f1ee", MUTED)
arrow(8.2, 2.4, 8.75, 3.05); arrow(8.2, 2.0, 8.75, 1.4)
save(fig, "figure1_framework")

# Figure 2: (a) ROC (b) temporal AUC per model
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
yy = y.values
for key, col, ls in [("GBDT | Base", C_BASE, "--"), ("KEXS (GBDT+K+mono)", C_KE, "-")]:
    fpr, tpr, _ = roc_curve(yy, oof_rep1[key])
    axs[0].plot(fpr, tpr, color=col, ls=ls, lw=2, label=f"{'GBDT (Base)' if 'Base' in key else 'KEXS'}  AUC={roc_auc_score(yy, oof_rep1[key]):.3f}")
axs[0].plot([0, 1], [0, 1], color=GRID, lw=1)
axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) Cross-validated ROC")
axs[0].legend(loc="lower right", fontsize=7.5)
models = ["LR", "MLP", "RF", "GBDT"]
xb = np.arange(len(models)); w = 0.36
base_t = [roc_auc_score(yte, temporal[f"{m} | Base"]) for m in models]
ke_t = [roc_auc_score(yte, temporal[f"{m} | Base+K"]) for m in models]
axs[1].bar(xb - w / 2 - 0.01, base_t, w, color=C_BASE, label="Base features")
axs[1].bar(xb + w / 2 + 0.01, ke_t, w, color=C_KE, label="Base + knowledge (K1–K4)", hatch="////", edgecolor="white", lw=0)
for i, (a, b) in enumerate(zip(base_t, ke_t)):
    axs[1].text(i + w / 2, max(a, b) + 0.004, f"{b - a:+.3f}", ha="center", fontsize=7, color=INK)
axs[1].set_xticks(xb, models); axs[1].set_ylim(0.6, max(ke_t + base_t) + 0.04)
axs[1].set(ylabel="ROC-AUC (2010–2014 cohort)", title="(b) Temporal hold-out")
axs[1].legend(loc="upper left", fontsize=7.5); axs[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "figure2_performance")

# Figure 3: group Shapley + leave-one-group-out
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
order = list(G)
vals = [phi[g] for g in order]
axs[0].barh(order[::-1], vals[::-1], color=[C_BASE if g == "Base" else C_KE for g in order[::-1]], height=0.6)
for i, v in enumerate(vals[::-1]):
    axs[0].text(v + 0.002, i, f"{v:.3f} ({100 * v / (v_all - 0.5):.0f}%)", va="center", fontsize=7, color=INK)
axs[0].set(xlabel="Shapley value (AUC points above 0.5)", title="(a) Group Shapley attribution")
axs[0].set_xlim(0, max(vals) * 1.45); axs[0].grid(axis="y", visible=False)
lv = [lofo[g] for g in order]
axs[1].barh(order[::-1], lv[::-1], color=[C_BASE if g == "Base" else C_KE for g in order[::-1]], height=0.6)
for i, v in enumerate(lv[::-1]):
    axs[1].text(max(v, 0) + 0.0003, i, f"{v:+.4f}", va="center", fontsize=7, color=INK)
axs[1].axvline(0, color=MUTED, lw=0.8)
axs[1].set(xlabel="AUC lost when group removed", title="(b) Leave-one-group-out")
axs[1].set_xlim(min(0, min(lv)) - 0.001, max(lv) * 1.5); axs[1].set_yticklabels([]); axs[1].grid(axis="y", visible=False)
fig.tight_layout(); save(fig, "figure3_attribution")

# Figure 4: (a) individual-level monotonicity violations, (b-d) partial dependence of KEXS
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
cf = ["k3_max_stage", "log_funding_total", "funding_rounds"]
cl = ["Stage", "Total\nfunding", "Rounds"]
xb = np.arange(len(cf)); w = 0.38
a = axs[0]
v0 = [100 * ice[f]["no_K5"] for f in cf]; v1 = [100 * ice[f]["KEXS"] for f in cf]
a.bar(xb - w / 2 - 0.01, v0, w, color=C_BASE, label="GBDT+K1–K4 (no K5)")
a.bar(xb + w / 2 + 0.01, v1, w, color=C_KE, label="KEXS (K5)", hatch="////", edgecolor="white", lw=0)
for i, v in enumerate(v0):
    a.text(i - w / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=6.5, color=INK)
for i, v in enumerate(v1):
    a.text(i + w / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=6.5, color=INK)
a.set_xticks(xb, cl, fontsize=7); a.set_ylim(0, max(v0) * 1.35); a.grid(axis="x", visible=False)
a.set_ylabel("Firms with a non-monotone\nindividual prediction (%)", fontsize=7.5)
a.set_title("(a) K5 guarantee", fontsize=8.5); a.legend(fontsize=6, loc="upper left")
for a, (f, lab), t in zip(axs[1:], pd_feats, ["(b)", "(c)", "(d)"]):
    gx, gy = pdp[f]
    a.plot(gx, gy, color=C_KE, lw=2, marker="o" if len(gx) <= 8 else None, ms=4)
    a.set_xlabel(lab, fontsize=7); a.set_title(f"{t} Partial dependence", fontsize=8.5)
axs[1].set_ylabel("Mean log-odds of exit", fontsize=7.5)
fig.tight_layout(); save(fig, "figure4_interpretability")

# ---------------- metrics.json ---------------------------------------------
metrics = {
    "n_labelled": int(n), "n_acquired": int(y.sum()), "n_closed": int((1 - y).sum()),
    "n_reference_firms": int(len(d_all)), "n_features": {g: len(c) for g, c in G.items()},
    "temporal": {"n_train": int(len(tr_t)), "n_test": int(len(te_t)),
                 "test_positive_rate": float(yte.mean())},
    "cv": {k: {m: [float(np.mean(v)), float(np.std(v))] for m, v in sc.items()} for k, sc in fold_scores.items()},
    "temporal_auc": {k: float(roc_auc_score(yte, p)) for k, p in temporal.items()},
    "group_shapley": phi, "v_all": v_all, "lofo": lofo,
    "subset_auc": {"+".join(sorted(k)) or "none": v for k, v in cache.items()},
    "pdp": pdp, "pdp_no_k5": pdp_free, "ice_violation_rate": ice,
}
with open(f"{OUT}/metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=1)
print("done.")
