#!/usr/bin/env python3
"""Shared primitives for the two calibration experiments (oracle + scalar bias).

Model-agnostic on purpose: everything here consumes a `logit_lens_per_sample_<method>.npz`
written by either `mech_interp/extract_cd_generic.py` (LLaVA-1.5-7B, 33 hidden states) or
`mech_interp_qwen/extract_cd.py` (Qwen2.5-VL-7B, 29 hidden states), and never touches a model.
The npz contract is:

    clean_margin    [n, L]  float32   expert logit-lens Yes-No margin per layer
    amateur_margin  [n, L]  float32   amateur logit-lens Yes-No margin per layer
    gt              [n]     int       1 = object present (POPE answer "yes"), 0 = absent
    pred            [n]     int       1 = model answered Yes, 0 = No
    split           [n]     str       random | popular | adversarial

Layer index -1 is the readout: the hidden state after the final norm, whose margin reproduces
the true decision (extract_summary_*.json records the lens MAE, 0.0 for the Qwen run).
"""
import numpy as np

READOUT = -1


def rank_auc(scores, labels):
    """AUC that `scores` ranks the `labels==True` class above the rest, ties averaged.

    Same implementation the layerwise battery uses (`analyze_all.py`), copied rather than
    imported so this package stays independent of both model-specific packages.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ss = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


class Lens:
    """One method's per-sample logit-lens margins plus the derived populations."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        self.path = path
        self.clean = z["clean_margin"].astype(np.float64)
        self.amateur = z["amateur_margin"].astype(np.float64)
        self.gt = z["gt"].astype(int)
        self.pred = z["pred"].astype(int)
        self.split = z["split"].astype(str)
        self.delta = self.clean - self.amateur
        self.n, self.n_layers = self.clean.shape
        # H = hallucinations (object absent, model says Yes); T = correct positives.
        self.H = (self.gt == 0) & (self.pred == 1)
        self.T = (self.gt == 1) & (self.pred == 1)
        self.pos = self.H | self.T
        # Label vector restricted to H u T: 1 on hallucinations, the class a selective CD
        # would have to suppress.
        self.y_H = self.H[self.pos].astype(int)

    @property
    def n_H(self):
        return int(self.H.sum())

    @property
    def n_T(self):
        return int(self.T.sum())

    def selectivity(self, layer=READOUT):
        """AUC with which -Delta ranks H above T. 0.5 = the shift cannot tell them apart."""
        return rank_auc(-self.delta[self.pos, layer], self.y_H)

    def selectivity_by_layer(self):
        return np.array([self.selectivity(l) for l in range(self.n_layers)])

    def delta_pos(self, layer=READOUT):
        return self.delta[self.pos, layer]

    def margin_pos(self, layer=READOUT):
        return self.clean[self.pos, layer]


def bootstrap_auc_ci(scores, labels, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap over samples for the selectivity AUC.

    Returns (lo, hi, se). Resampling is stratified within the two classes so every replicate
    keeps |H| and |T| fixed; with |H| two orders of magnitude below |T| an unstratified
    resample would occasionally produce a degenerate replicate.
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels).astype(bool)
    idx_pos = np.flatnonzero(labels)
    idx_neg = np.flatnonzero(~labels)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([rng.choice(idx_pos, idx_pos.size, replace=True),
                               rng.choice(idx_neg, idx_neg.size, replace=True)])
        boot[b] = rank_auc(scores[take], labels[take])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(boot.std(ddof=1))


def probe_oof_proba(X, y, folds=5, seed=0):
    """Out-of-fold P(y=1) from a standardized logistic probe, stratified k-fold.

    Out-of-fold is load-bearing: the oracle built from these probabilities is meant to be an
    upper bound on a *detectable* selective shift, not a memorized one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def pope_scores(pred_yes, gt):
    """POPE accuracy / F1 / precision / recall / yes-rate for a boolean Yes-decision vector.

    POPE's positive class is "yes" (object present), matching `eval/pope_eval_base.py`.
    """
    pred_yes = np.asarray(pred_yes).astype(bool)
    truth = np.asarray(gt).astype(bool)
    tp = int((pred_yes & truth).sum())
    fp = int((pred_yes & ~truth).sum())
    fn = int((~pred_yes & truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "accuracy": float((pred_yes == truth).mean()),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "yes_rate": float(pred_yes.mean()),
    }


MODEL_LABELS = {
    "llava1.5-7b": "LLaVA-1.5-7B",
    "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
}


def model_label(tag):
    return MODEL_LABELS.get(tag, tag)
