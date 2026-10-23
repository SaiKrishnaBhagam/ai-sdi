"""KEXS: Knowledge-Enhanced eXplainable Startup-success model.

Core library: data preparation, the four knowledge layers (K1-K4),
model factory with knowledge-based monotone constraints (K5), and
exact feature-group Shapley attribution.

Everything is deterministic given `SEED`.
"""
from __future__ import annotations

import re
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SEED = 42

FUND_TYPES = ["seed", "venture", "equity_crowdfunding", "undisclosed", "convertible_note",
              "debt_financing", "angel", "grant", "private_equity", "post_ipo_equity",
              "post_ipo_debt", "secondary_market", "product_crowdfunding",
              "round_A", "round_B", "round_C", "round_D", "round_E", "round_F",
              "round_G", "round_H"]

# ---------------------------------------------------------------------------
# K1  Sector ontology: expert taxonomy mapping 800+ Crunchbase categories to
#     10 macro-sectors, each annotated with a capital-intensity level
#     (1 = asset-light, 3 = capital-heavy), following the VC literature.
# ---------------------------------------------------------------------------
ONTOLOGY = {
    # sector: (capital_intensity, keyword list)
    "life_sciences": (3, ["biotech", "pharma", "health", "medical", "therapeut", "diagnost",
                          "life science", "drug", "hospital", "genetic", "fitness", "wellness",
                          "dental", "nutrition", "bio", "doctor", "care"]),
    "clean_energy":  (3, ["clean", "energy", "solar", "renewable", "battery", "green",
                          "environment", "water", "oil", "gas", "fuel", "efficiency", "recycl"]),
    "deep_hardware": (3, ["semiconductor", "hardware", "manufactur", "electronic", "robot",
                          "device", "material", "nanotech", "aerospace", "automotive",
                          "3d", "sensor", "optical", "chip", "industrial", "machine"]),
    "fintech":       (2, ["financ", "bank", "payment", "insur", "invest", "lending", "credit",
                          "bitcoin", "accounting", "trading", "real estate", "crowdfund"]),
    "telecom_mobile": (2, ["mobile", "telecom", "wireless", "network", "app", "messaging",
                           "communications", "voip", "internet of things"]),
    "enterprise":    (1, ["enterprise", "saas", "analytics", "big data", "security", "cloud",
                          "software", "b2b", "crm", "business", "consult", "hosting",
                          "infrastructure", "database", "developer", "human resources",
                          "recruit", "logistics", "legal", "marketing", "advertis", "sales"]),
    "consumer_commerce": (1, ["e-commerce", "commerce", "retail", "fashion", "food",
                              "restaurant", "consumer", "shopping", "marketplace", "travel",
                              "hospitality", "local", "lifestyle", "beauty", "home", "pet",
                              "kids", "wedding", "event", "transport"]),
    "media_content": (1, ["media", "game", "gaming", "music", "video", "entertainment",
                          "news", "publishing", "photo", "social", "content", "sports",
                          "television", "film", "art", "design", "blog", "broadcast", "concert", "virtual world"]),
    "education":     (1, ["educat", "learning", "school", "university", "training", "tutor"]),
    "internet_web":  (1, ["web", "internet", "search", "online", "curated", "platform",
                          "technology", "open source", "email", "browser"]),
}
SECTORS = list(ONTOLOGY)


def _parse_money(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s.replace({"-": np.nan, "nan": np.nan}), errors="coerce")


def load_raw(path: str) -> pd.DataFrame:
    """Load the Crunchbase export; returns ALL firms (labelled + unlabelled)."""
    d = pd.read_csv(path, encoding="latin-1")
    d.columns = [c.strip() for c in d.columns]
    d = d.dropna(subset=["permalink"]).drop_duplicates("permalink").reset_index(drop=True)
    d["funding_total_usd"] = _parse_money(d["funding_total_usd"])
    for c in ["founded_at", "first_funding_at", "last_funding_at"]:
        d[c] = pd.to_datetime(d[c], errors="coerce")
    for c in ["first_funding_at", "last_funding_at", "founded_at"]:   # remove impossible dates
        d.loc[(d[c].dt.year < 1900) | (d[c].dt.year > 2015), c] = pd.NaT
    d["market"] = d["market"].astype(str).str.strip().replace("nan", np.nan)
    d["cats"] = (d["category_list"].fillna("").str.strip("|").str.lower()
                 .str.split("|").apply(lambda x: [c.strip() for c in x if c.strip()]))
    for c in ["country_code", "region", "city"]:
        d[c] = d[c].fillna("UNK").astype(str).str.strip()
    return d


def label(d: pd.DataFrame) -> pd.Series:
    """Success = acquired (1) vs failure = closed (0); operating firms are unresolved (NaN)."""
    return d["status"].map({"acquired": 1, "closed": 0})


# ---------------------------------------------------------------------------
# Base (data-only) features: what a standard ML pipeline would use.
# ---------------------------------------------------------------------------
def base_features(d: pd.DataFrame, top_markets: list[str], top_countries: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["log_funding_total"] = np.log1p(d["funding_total_usd"])
    X["funding_rounds"] = d["funding_rounds"]
    for f in FUND_TYPES:
        X[f"log_{f}"] = np.log1p(d[f].fillna(0))
    X["founded_year"] = d["founded_year"]
    X["first_funding_year"] = d["first_funding_at"].dt.year
    for m in top_markets:
        X[f"mkt_{m}"] = (d["market"] == m).astype(int)
    for c in top_countries:
        X[f"cty_{c}"] = (d["country_code"] == c).astype(int)
    return X


# ---------------------------------------------------------------------------
# Knowledge layers
# ---------------------------------------------------------------------------
# Explicit expert assignments for categories the keyword rules would miss or confuse.
OVERRIDES = {
    "telecom_mobile": ["iphone", "android", "ios", "ipad", "tablets", "sms", "chat",
                       "location based services", "maps", "network security", "gps",
                       "tracking", "mobility", "facebook applications", "twitter applications"],
    "enterprise": ["collaboration", "project management", "customer service", "lead generation",
                   "seo", "public relations", "paas", "virtualization", "storage",
                   "data visualization", "optimization", "identity", "ediscovery",
                   "personalization", "real time", "employment", "machine learning",
                   "artificial intelligence", "market research", "career management",
                   "ad targeting", "promotional", "monetization", "outsourcing", "freelancers",
                   "data integration", "information services", "privacy", "distribution"],
    "consumer_commerce": ["reviews and recommendations", "coupons", "loyalty programs", "discounts",
                          "hotels", "hospitality", "ticketing", "gift card", "auctions",
                          "price comparison", "cars", "parenting", "collaborative consumption",
                          "peer-to-peer", "incentives", "tourism", "pets", "creative"],
    "media_content": ["audio", "graphics", "gamification", "communities", "crowdsourcing",
                      "file sharing"],
    "education": ["colleges", "all students"],
    "fintech": ["venture capital", "point of sale"],
    "clean_energy": ["smart grid"],
}
_OVR = {c: s for s, cs in OVERRIDES.items() for c in cs}
_PATTERNS = {s: [re.compile(r"\b" + re.escape(k) + (r"(s|es|ing)?\b" if len(k) <= 4 else ""))
                 for k in kws] for s, (_, kws) in ONTOLOGY.items()}


def sector_of(cat: str):
    """Ontology lookup: explicit override, else first sector whose keyword matches."""
    if cat in _OVR:
        return _OVR[cat]
    for s, pats in _PATTERNS.items():
        if any(p.search(cat) for p in pats):
            return s
    return None


def k1_ontology(d: pd.DataFrame) -> pd.DataFrame:
    """K1: map multi-label categories onto the expert sector ontology."""
    lut = {c: sector_of(c) for c in {c for cs in d["cats"] for c in cs}}
    hits = d["cats"].apply(lambda cs: {lut[c] for c in cs if lut[c]})
    X = pd.DataFrame(index=d.index)
    for s in SECTORS:
        X[f"k1_{s}"] = hits.apply(lambda h: int(s in h))
    X["k1_sector_breadth"] = hits.apply(len)
    X["k1_capital_intensity"] = hits.apply(lambda h: max([ONTOLOGY[s][0] for s in h], default=np.nan))
    X["k1_n_categories"] = d["cats"].apply(len)
    return X


def k2_category_graph(d_all: pd.DataFrame, d: pd.DataFrame, dim: int = 16) -> pd.DataFrame:
    """K2: category co-occurrence knowledge graph -> PPMI -> spectral embedding.

    Built from all firms (label-free, includes unresolved 'operating' firms),
    so no outcome information enters the graph.
    """
    vocab = {c: i for i, c in enumerate(sorted({c for cs in d_all["cats"] for c in cs}))}
    rows, cols = [], []
    for r, cs in enumerate(d_all["cats"]):
        for c in cs:
            rows.append(r); cols.append(vocab[c])
    B = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(d_all), len(vocab)))
    C = (B.T @ B).toarray().astype(float)               # category co-occurrence graph
    np.fill_diagonal(C, 0)
    tot = C.sum(); rs = C.sum(1, keepdims=True); cs_ = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * tot) / (rs @ cs_))
    ppmi = np.nan_to_num(np.maximum(pmi, 0), nan=0.0, posinf=0.0)
    emb = TruncatedSVD(n_components=dim, random_state=SEED).fit_transform(ppmi)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    out = np.zeros((len(d), dim))
    for i, cs in enumerate(d["cats"]):
        idx = [vocab[c] for c in cs if c in vocab]
        if idx:
            out[i] = emb[idx].mean(0)
    return pd.DataFrame(out, index=d.index, columns=[f"k2_emb{j}" for j in range(dim)])


def k3_financing(d: pd.DataFrame) -> pd.DataFrame:
    """K3: venture-staging knowledge (stage progression, investor mix, tempo)."""
    X = pd.DataFrame(index=d.index)
    stage = np.zeros(len(d))
    stage[((d["seed"] > 0) | (d["angel"] > 0)).values] = 1
    for lvl, cols in [(2, ["round_A"]), (3, ["round_B"]), (4, ["round_C"]),
                      (5, ["round_D", "round_E", "round_F", "round_G", "round_H"])]:
        stage[(d[cols].fillna(0).sum(1) > 0).values] = lvl
    X["k3_max_stage"] = stage
    total = d[FUND_TYPES].fillna(0).sum(1).replace(0, np.nan)
    inst = d[["venture", "private_equity"] + [c for c in FUND_TYPES if c.startswith("round_")]].fillna(0).sum(1)
    X["k3_institutional_share"] = inst / total
    X["k3_informal_share"] = d[["seed", "angel", "equity_crowdfunding", "product_crowdfunding"]].fillna(0).sum(1) / total
    X["k3_nondilutive_share"] = d[["grant", "debt_financing"]].fillna(0).sum(1) / total
    X["k3_n_instrument_types"] = (d[FUND_TYPES].fillna(0) > 0).sum(1)
    span = (d["last_funding_at"] - d["first_funding_at"]).dt.days / 365.25
    X["k3_funding_span_yrs"] = span
    X["k3_round_velocity"] = d["funding_rounds"] / (span.clip(lower=0) + 1)
    X["k3_time_to_first_funding"] = ((d["first_funding_at"] - d["founded_at"]).dt.days / 365.25).clip(0, 30)
    X["k3_log_funding_per_round"] = np.log1p(d["funding_total_usd"] / d["funding_rounds"])
    return X


def k4_ecosystem(d_all: pd.DataFrame, d: pd.DataFrame, sect_all: pd.DataFrame, sect: pd.DataFrame) -> pd.DataFrame:
    """K4: entrepreneurial-ecosystem (agglomeration) knowledge, label-free."""
    X = pd.DataFrame(index=d.index)
    reg_n = d_all["region"].value_counts()
    reg_f = d_all.groupby("region")["funding_total_usd"].sum()
    cty_n = d_all["country_code"].value_counts()
    mkt_n = d_all["market"].value_counts()
    X["k4_log_region_density"] = np.log1p(d["region"].map(reg_n).fillna(0))
    X["k4_log_region_capital"] = np.log1p(d["region"].map(reg_f).fillna(0))
    X["k4_log_country_density"] = np.log1p(d["country_code"].map(cty_n).fillna(0))
    X["k4_log_market_crowding"] = np.log1p(d["market"].map(mkt_n).fillna(0))
    hubs = set(reg_n.drop("UNK", errors="ignore").index[:10])
    X["k4_top10_hub"] = d["region"].isin(hubs).astype(int)
    # sector-in-region specialisation (location quotient)
    s_cols = [f"k1_{s}" for s in SECTORS]
    tmp = sect_all[s_cols].copy(); tmp["region"] = d_all["region"].values
    reg_sec = tmp.groupby("region")[s_cols].sum()
    reg_tot = tmp.groupby("region").size()
    nat = sect_all[s_cols].sum() / len(sect_all)
    lq = (reg_sec.div(reg_tot, axis=0)) / nat.replace(0, np.nan)
    lq_rows = lq.reindex(d["region"]).values                     # region x sector LQ
    own = sect[s_cols].values
    with np.errstate(invalid="ignore"):
        X["k4_sector_location_quotient"] = np.where(own.sum(1) > 0,
                                                    np.nansum(lq_rows * own, 1) / np.maximum(own.sum(1), 1),
                                                    np.nan)
    return X


GROUPS = ["Base", "K1 ontology", "K2 category graph", "K3 financing", "K4 ecosystem"]
PREFIX = {"K1 ontology": "k1_", "K2 category graph": "k2_", "K3 financing": "k3_", "K4 ecosystem": "k4_"}


def build(path: str, ref_max_year: int | None = None):
    """Returns X (all features), y, groups{name: cols}, meta, d_all for labelled firms.

    The label-free knowledge structures (category graph K2, ecosystem statistics K4,
    vocabularies of the Base encodings) are built from a *reference* population.
    By default this is every firm in the export; with `ref_max_year` only firms whose
    first funding happened in or before that year are used (strict temporal setting).
    """
    d_all = load_raw(path)
    if ref_max_year is None:
        ref = d_all
    else:
        ref = d_all[d_all["first_funding_at"].dt.year.fillna(9999) <= ref_max_year].reset_index(drop=True)
    top_markets = ref["market"].value_counts().index[:30].tolist()
    top_countries = ref["country_code"].value_counts().drop("UNK", errors="ignore").index[:15].tolist()
    k1_all, k1_ref = k1_ontology(d_all), k1_ontology(ref)
    feats = pd.concat([base_features(d_all, top_markets, top_countries), k1_all,
                       k2_category_graph(ref, d_all), k3_financing(d_all),
                       k4_ecosystem(ref, d_all, k1_ref, k1_all)], axis=1)
    y = label(d_all)
    keep = y.notna()
    X = feats[keep].reset_index(drop=True).astype(float)
    meta = d_all.loc[keep, ["permalink", "market", "country_code", "first_funding_at", "founded_at"]].reset_index(drop=True)
    groups = {"Base": [c for c in X.columns if not c.startswith(("k1_", "k2_", "k3_", "k4_"))]}
    groups.update({g: [c for c in X.columns if c.startswith(p)] for g, p in PREFIX.items()})
    return X, y[keep].astype(int).reset_index(drop=True), groups, meta, d_all


# ---------------------------------------------------------------------------
# K5  knowledge-based monotone constraints (prior: more advanced staging,
#     more institutional capital, more capital -> higher exit likelihood).
# ---------------------------------------------------------------------------
MONOTONE_UP = ["k3_max_stage", "k3_institutional_share", "log_funding_total", "funding_rounds"]


def make_model(name: str, cols: list[str], monotone: bool = False):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=3000))
    if name == "MLP":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                           max_iter=500, random_state=SEED))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=400, min_samples_leaf=3, max_features="sqrt",
                                                    n_jobs=-1, random_state=SEED))
    if name == "GBDT":
        cst = [1 if (monotone and c in MONOTONE_UP) else 0 for c in cols]
        # hyper-parameters selected on the Base feature set only (5-fold CV,
        # separate seed), then frozen for every feature set -> conservative test.
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=7,
                                              min_samples_leaf=80, l2_regularization=5.0,
                                              monotonic_cst=cst, early_stopping=False,
                                              random_state=SEED)
    raise ValueError(name)


def cv_auc(X: pd.DataFrame, y: pd.Series, cols: list[str], folds, model="GBDT", monotone=False) -> float:
    if not cols:
        return 0.5
    oof = np.zeros(len(y))
    for tr, te in folds:
        m = make_model(model, cols, monotone).fit(X.iloc[tr][cols], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
    return roc_auc_score(y, oof)


def group_shapley(X, y, groups: dict, folds, model="GBDT") -> dict:
    """Exact Shapley value of each feature group w.r.t. cross-validated AUC.

    v(S) = OOF AUC of a model trained on the union of groups in S, v({}) = 0.5.
    """
    names = list(groups)
    n = len(names)
    cache = {}
    def v(S):
        key = frozenset(S)
        if key not in cache:
            cols = [c for g in S for c in groups[g]]
            cache[key] = cv_auc(X, y, cols, folds, model)
        return cache[key]
    phi = {}
    for g in names:
        others = [h for h in names if h != g]
        tot = 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for S in combinations(others, k):
                tot += w * (v(S + (g,)) - v(S))
        phi[g] = tot
    return phi, cache
