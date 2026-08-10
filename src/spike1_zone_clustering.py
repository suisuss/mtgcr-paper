"""Spike 1: shared turn-point clustering (zone-risk H1 descriptive arm,
endpoint 2a geography, Bar 2 baseline head, leakage-controlled zone-dwell
comparator). Protocol fixed in PREREGISTRATION.md before any run.

    python spike1_zone_clustering.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import beta
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from lib.campaign import save_result
from lib.actr_data import load_uss, band

K_PRIMARY = 64
K_SENS = 16
SEED = 42
SEED_SENS = 123
N_BOOT = 1000
TAU_BINS = [0.0, 0.25, 0.5, 0.75, 1.000001]


def _flatten(convs):
    """Turn-level arrays across a corpus: X, conv_id, tau, is_user."""
    X, cid, tau, usr = [], [], [], []
    for i, c in enumerate(convs):
        n = len(c["emb"])
        X.append(c["emb"])
        cid.append(np.full(n, i))
        tau.append(np.arange(n) / max(n - 1, 1))
        usr.append(c["roles"] == "USER")
    return (np.vstack(X), np.concatenate(cid), np.concatenate(tau),
            np.concatenate(usr))


def _lift_table(assign, cid, bands, k):
    """Per-cluster bad-turn share vs base rate over band turns.

    Returns (per-conv count matrix T [n_band_convs, k], bad-conv mask,
    shares, lifts, jeffreys CIs, base rate).
    """
    band_convs = [i for i, b in bands.items() if b is not None]
    idx_of = {c: j for j, c in enumerate(band_convs)}
    y = np.array([bands[c] for c in band_convs], dtype=bool)
    T = np.zeros((len(band_convs), k))
    mask = np.isin(cid, band_convs)
    for c, a in zip(cid[mask], assign[mask]):
        T[idx_of[c], a] += 1
    tot = T.sum(0)
    bad = T[y].sum(0)
    base = bad.sum() / tot.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(tot > 0, bad / np.maximum(tot, 1), np.nan)
        lift = share / base
    lo = beta.ppf(0.025, bad + 0.5, tot - bad + 0.5)
    hi = beta.ppf(0.975, bad + 0.5, tot - bad + 0.5)
    return T, y, share, lift, (lo / base, hi / base), base


def _bootstrap_lift(T, y, rng):
    """Conversation-level bootstrap CI of per-cluster lift."""
    n, k = T.shape
    draws = rng.integers(0, n, size=(N_BOOT, n))
    lifts = np.full((N_BOOT, k), np.nan)
    for b in range(N_BOOT):
        idx = draws[b]
        tot = T[idx].sum(0)
        bad = T[idx][y[idx]].sum(0)
        base = bad.sum() / max(tot.sum(), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            lifts[b] = np.where(tot > 0, bad / np.maximum(tot, 1), np.nan) / base
    return (np.nanpercentile(lifts, 2.5, axis=0),
            np.nanpercentile(lifts, 97.5, axis=0))


def _high_risk(lift, boot_lo):
    return np.where((lift >= 2.0) & (boot_lo > 1.0))[0]


def _prefix_samples(convs, bands):
    """One sample per user-turn prefix of every band conversation."""
    feats, labels, taus, groups, hr_dwell_raw = [], [], [], [], []
    for i, c in enumerate(convs):
        b = bands[i]
        if b is None:
            continue
        emb, roles = c["emb"], c["roles"]
        n = len(emb)
        for t in range(n):
            if roles[t] != "USER":
                continue
            u = emb[:t + 1][roles[:t + 1] == "USER"]
            s = emb[:t + 1][roles[:t + 1] == "SYSTEM"]
            um = u.mean(0)
            sm = s.mean(0) if len(s) else np.zeros(emb.shape[1])
            feats.append(np.concatenate([um, sm]))
            labels.append(b)
            taus.append(t / max(n - 1, 1))
            groups.append(i)
    return (np.array(feats), np.array(labels), np.array(taus),
            np.array(groups))


def _auc_by_tau(y, score, tau):
    out = []
    for lo, hi in zip(TAU_BINS[:-1], TAU_BINS[1:]):
        m = (tau >= lo) & (tau < hi)
        if m.sum() and len(set(y[m])) == 2:
            out.append(float(roc_auc_score(y[m], score[m])))
        else:
            out.append(None)
    return out


def _endpoint_geography(assign, cid, bands, convs, k, rng):
    """TV distance between bad/good final-turn cluster distributions,
    with a conversation-label permutation null (reporting summary for the
    pre-registered by-band distribution)."""
    finals = {}
    for i in range(len(convs)):
        m = np.where(cid == i)[0]
        finals[i] = assign[m[-1]]
    band_convs = [i for i, b in bands.items() if b is not None]
    y = np.array([bands[i] for i in band_convs], dtype=bool)
    f = np.array([finals[i] for i in band_convs])

    def tv(labels):
        p = np.bincount(f[labels], minlength=k) / max(labels.sum(), 1)
        q = np.bincount(f[~labels], minlength=k) / max((~labels).sum(), 1)
        return 0.5 * np.abs(p - q).sum()

    obs = tv(y)
    perms = np.array([tv(rng.permutation(y)) for _ in range(N_BOOT)])
    return {"tv": float(obs), "perm_mean": float(perms.mean()),
            "perm_p95": float(np.percentile(perms, 95)),
            "perm_pvalue": float((perms >= obs).mean())}


def _lead_time(assign, cid, tau, convs, bands, hr):
    """First entry of bad-band conversations into a high-risk cluster;
    'early' = strictly before the second-to-last user turn."""
    hr = set(hr.tolist())
    rows = []
    for i, c in enumerate(convs):
        if bands[i] != 1:
            continue
        m = np.where(cid == i)[0]
        in_hr = [j for j in m if assign[j] in hr]
        if not in_hr:
            continue
        first = in_hr[0]
        user_pos = m[convs[i]["roles"] == "USER"]
        cutoff = user_pos[-2] if len(user_pos) >= 2 else user_pos[-1]
        rows.append({"first_tau": float(tau[first]),
                     "early": bool(first < cutoff)})
    if not rows:
        return {"n_entering": 0, "median_first_tau": None, "early_share": None}
    return {"n_entering": len(rows),
            "median_first_tau": float(np.median([r["first_tau"] for r in rows])),
            "early_share": float(np.mean([r["early"] for r in rows]))}


def _fold_dwell_scores(X, cid, assign, samples, k):
    """Leakage-controlled dwell: per CV fold, high-risk clusters are
    ranked on the train fold only (lift >= 2 with the Jeffreys share CI
    lower bound above 1x train base), then dwell (share of prefix turns
    in those clusters) scores the test-fold samples."""
    feats, labels, taus, groups = samples
    dwell = np.full(len(labels), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(feats, labels, groups):
        tr_convs = set(groups[tr].tolist())
        bands_tr = {c: bool(labels[tr][groups[tr] == c][0]) for c in tr_convs}
        T, y, share, lift, (jlo, _), base = _lift_table(
            assign, cid, {**{c: None for c in set(cid.tolist())}, **bands_tr}, k)
        hr = set(np.where((lift >= 2.0) & (jlo > 1.0))[0].tolist())
        for s in te:
            conv, t_tau = groups[s], taus[s]
            m = np.where(cid == conv)[0]
            n = len(m)
            upto = m[:int(round(t_tau * max(n - 1, 1))) + 1]
            dwell[s] = np.mean([assign[j] in hr for j in upto]) if len(upto) else 0.0
    return dwell


def run(corpora=("sgd", "mwoz")):
    rng = np.random.default_rng(SEED)
    all_results = {}
    for corpus in corpora:
        convs = load_uss(corpus)
        bands = {i: band(c) for i, c in enumerate(convs)}
        X, cid, tau, usr = _flatten(convs)

        km = KMeans(n_clusters=K_PRIMARY, n_init=10, random_state=SEED)
        assign = km.fit_predict(X)

        T, y, share, lift, (jlo, jhi), base = _lift_table(
            assign, cid, bands, K_PRIMARY)
        blo, bhi = _bootstrap_lift(T, y, rng)
        hr = _high_risk(lift, blo)

        order = np.argsort(np.nan_to_num(lift, nan=-1))[::-1][:5]
        top = [{"cluster": int(j), "n_band_turns": int(T[:, j].sum()),
                "share": float(share[j]), "lift": float(lift[j]),
                "jeffreys": [float(jlo[j]), float(jhi[j])],
                "boot_ci": [float(blo[j]), float(bhi[j])]} for j in order]

        geo = _endpoint_geography(assign, cid, bands, convs, K_PRIMARY, rng)
        lead = _lead_time(assign, cid, tau, convs, bands, hr)

        samples = _prefix_samples(convs, bands)
        feats, labels, staus, groups = samples
        head_oof = np.full(len(labels), np.nan)
        gkf = GroupKFold(n_splits=5)
        for tr, te in gkf.split(feats, labels, groups):
            clf = LogisticRegression(C=1.0, class_weight="balanced",
                                     max_iter=2000)
            clf.fit(feats[tr], labels[tr])
            head_oof[te] = clf.predict_proba(feats[te])[:, 1]
        dwell = _fold_dwell_scores(X, cid, assign, samples, K_PRIMARY)

        sens = {}
        for tag, kk, sd in (("k16", K_SENS, SEED), ("seed123", K_PRIMARY,
                                                    SEED_SENS)):
            a2 = KMeans(n_clusters=kk, n_init=10,
                        random_state=sd).fit_predict(X)
            T2, y2, _, l2, (j2lo, _), _ = _lift_table(a2, cid, bands, kk)
            b2lo, _ = _bootstrap_lift(T2, y2, rng)
            h2 = _high_risk(l2, b2lo)
            sens[tag] = {"max_lift": float(np.nanmax(l2)),
                         "n_highrisk": int(len(h2))}

        res = {
            "corpus": corpus, "k": K_PRIMARY, "seed": SEED,
            "n_conv": len(convs),
            "n_bad": int(sum(1 for b in bands.values() if b == 1)),
            "n_good": int(sum(1 for b in bands.values() if b == 0)),
            "base_rate": float(base),
            "top_clusters": top,
            "n_highrisk": int(len(hr)),
            "max_lift": float(np.nanmax(lift)),
            "max_lift_boot_ci": [float(blo[order[0]]), float(bhi[order[0]])],
            "endpoint_geography": geo,
            "lead_time": lead,
            "auc_head_by_tau": _auc_by_tau(labels, head_oof, staus),
            "auc_dwell_by_tau": _auc_by_tau(labels, dwell, staus),
            "n_samples": int(len(labels)),
            "sensitivities": sens,
        }
        save_result(f"spike1_zone_{corpus}", res, dataset=corpus,
                    params={"k": K_PRIMARY, "seed": SEED, "n_boot": N_BOOT})
        all_results[corpus] = res
    return all_results


def main(argv=None):
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
