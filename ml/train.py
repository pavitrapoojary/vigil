"""Vigil training pipeline.
Loads VAERS DATA + SYMPTOMS, builds P1/P2/P3 labels from reporter outcome flags,
removes flags from model input (narrative only), trains TF-IDF + logistic regression,
evaluates on a held-out test set, and exports a JSON model for the JS serverless API.
Usage: python ml/train.py --data path/2025VAERSDATA.csv --symptoms path/2025VAERSSYMPTOMS.csv
"""
import argparse, json, re, time, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--symptoms", required=True)
ap.add_argument("--out", default="."); a = ap.parse_args()
SEED = 42; CLASSES = ["P1", "P2", "P3"]
here = os.path.dirname(os.path.abspath(__file__))

# ---------- 1. load, join, label ----------
d = pd.read_csv(a.data, encoding="latin-1", low_memory=False)
s = pd.read_csv(a.symptoms, encoding="latin-1")
flags = ["DIED", "L_THREAT", "HOSPITAL", "DISABLE"]
for f in flags: d[f] = (d[f] == "Y").astype(int)
# some VAERS_IDs appear on several rows (follow-up versions): merge, OR the flags, keep longest narrative
d["SYMPTOM_TEXT"] = d["SYMPTOM_TEXT"].fillna("")
d = (d.sort_values("SYMPTOM_TEXT", key=lambda x: x.str.len(), ascending=False)
       .groupby("VAERS_ID", as_index=False)
       .agg({**{f: "max" for f in flags}, "SYMPTOM_TEXT": "first", "AGE_YRS": "first", "SEX": "first"}))
pt_cols = [f"SYMPTOM{i}" for i in range(1, 6)]
pts = (s.melt(id_vars="VAERS_ID", value_vars=pt_cols).dropna(subset=["value"])
         .groupby("VAERS_ID")["value"].apply(lambda x: sorted(set(v.lower() for v in x))).rename("PTS"))
d = d.merge(pts, on="VAERS_ID", how="left")
d["PTS"] = d["PTS"].apply(lambda x: x if isinstance(x, list) else [])
d = d[d["SYMPTOM_TEXT"].str.len() >= 20].reset_index(drop=True)
d["label"] = np.where((d.DIED == 1) | (d.L_THREAT == 1), "P1",
              np.where((d.HOSPITAL == 1) | (d.DISABLE == 1), "P2", "P3"))
print("reports:", len(d)); print(d.label.value_counts())

# ---------- 2. split (flags are NOT model input) ----------
tr, tmp = train_test_split(d, test_size=0.30, stratify=d.label, random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.label, random_state=SEED)
vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=5, max_features=40000, sublinear_tf=True,
                      token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
Xtr = vec.fit_transform(tr.SYMPTOM_TEXT)
best = None
for C in [1, 4, 10]:
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Xtr, tr.label)
    pv = m.predict(vec.transform(va.SYMPTOM_TEXT))
    r = precision_recall_fscore_support(va.label, pv, labels=["P1", "P2"], zero_division=0)[1].mean()
    print(f"C={C} val serious-recall={r:.3f}")
    if best is None or r > best[0]: best = (r, C, m)
_, C, model = best; print("chosen C =", C)
assert list(model.classes_) == CLASSES

def proba(df): return model.predict_proba(vec.transform(df.SYMPTOM_TEXT))
t0 = time.time(); Pte = proba(te); ms_per_case = (time.time() - t0) / len(te) * 1000
pred = np.array(CLASSES)[Pte.argmax(1)]; y = te.label.values
rep = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
cm = confusion_matrix(y, pred, labels=CLASSES).tolist()
ser = np.isin(y, ["P1", "P2"]); ser_pred = np.isin(pred, ["P1", "P2"])
serious = {"recall": float((ser & ser_pred).sum() / ser.sum()), "precision": float((ser & ser_pred).sum() / max(ser_pred.sum(), 1)),
           "p1_recall_as_p1_or_p2": float(np.isin(pred[y == "P1"], ["P1", "P2"]).mean())}
print(classification_report(y, pred, labels=CLASSES, zero_division=0)); print("serious (P1|P2 vs P3):", serious)

# ---------- 3. subgroup performance ----------
te = te.assign(pred=pred, ser=ser, ser_pred=ser_pred)
te["age_group"] = pd.cut(te.AGE_YRS, [-1, 17, 49, 64, 200], labels=["0-17", "18-49", "50-64", "65+"]).astype(object).fillna("Unknown")
te["sex"] = te.SEX.map({"F": "Female", "M": "Male"}).fillna("Unknown")
def grp(col):
    out = []
    for k, g in te.groupby(col):
        n_ser = int(g.ser.sum())
        out.append({"group": k, "n": len(g), "n_serious": n_ser,
                    "serious_recall": float((g.ser & g.ser_pred).sum() / n_ser) if n_ser else None})
    return out

# ---------- 4. threshold trade-off (policy: auto-route if max prob >= tau, else manual) ----------
conf = Pte.max(1); curve = []
for tau in np.round(np.arange(0.30, 0.96, 0.05), 2):
    auto = conf >= tau
    missed = auto & ser & (pred == "P3")                  # serious case auto-routed to non-serious queue
    p1_under = auto & (y == "P1") & (pred != "P1")        # P1 not sent to P1 queue
    curve.append({"tau": float(tau), "auto_pct": float(auto.mean()), "manual_pct": float(1 - auto.mean()),
                  "serious_missed": int(missed.sum()), "serious_total": int(ser.sum()),
                  "serious_missed_pct": float(missed.sum() / ser.sum()), "p1_under_triaged": int(p1_under.sum()),
                  "p1_total": int((y == "P1").sum()), "auto_accuracy": float((pred[auto] == y[auto]).mean()) if auto.any() else None})
# default tau: highest auto-routing share that keeps serious_missed_pct <= 5%
ok = [c for c in curve if c["serious_missed_pct"] <= 0.05]
default_tau = max(ok, key=lambda c: c["auto_pct"])["tau"] if ok else 0.9
print("default tau:", default_tau)

# ---------- 5. validate AESI rules vs. coded MedDRA PTs ----------
rules = json.load(open(os.path.join(here, "rules.json")))
pt_map = {"anaphylaxis": r"anaphyla", "myocarditis": r"myocarditis|pericarditis|troponin", "gbs": r"guillain",
          "tts": r"thrombocytop|venous sinus|pulmonary embol|deep vein", "neuro": r"myelitis|encephalitis|stroke|seiz|convuls|cerebrovascular"}
neg = set(rules["negations"])
def rule_hit(text, pat):
    t = text.lower()
    for m in re.finditer(pat, t):
        before = re.findall(r"\w+", t[max(0, m.start() - 40):m.start()])[-4:]
        if not (set(before) & neg): return True
    return False
aesi_val = []
for r in rules["aesi"]:
    if r["id"] not in pt_map: continue
    pat = re.compile(pt_map[r["id"]]); truth = d.PTS.apply(lambda l: any(pat.search(x) for x in l))
    hit = d.SYMPTOM_TEXT.apply(lambda t: rule_hit(t, r["pattern"]))
    tp = int((truth & hit).sum())
    aesi_val.append({"id": r["id"], "label": r["label"], "coded_pt_cases": int(truth.sum()), "rule_hits": int(hit.sum()),
                     "recall_vs_coded": tp / max(int(truth.sum()), 1), "precision_vs_coded": tp / max(int(hit.sum()), 1)})
    print(aesi_val[-1])

# rules + model combined on test set (death/arrest rules force P1, AESI >= P2)
order = {"P1": 0, "P2": 1, "P3": 2}
def apply_rules(text, p):
    best = p
    for r in rules["aesi"]:
        if rule_hit(text, r["pattern"]) and order[r["min_priority"]] < order[best]: best = r["min_priority"]
    return best
comb = np.array([apply_rules(t, p) for t, p in zip(te.SYMPTOM_TEXT, pred)])
cser = np.isin(comb, ["P1", "P2"])
combined = {"serious_recall": float((ser & cser).sum() / ser.sum()), "serious_precision": float((ser & cser).sum() / cser.sum()),
            "p1_recall": float((comb[y == "P1"] == "P1").mean())}
print("model+rules:", combined)

# ---------- 6. error analysis ----------
fn = te[(te.ser) & (te.pred == "P3")]
fp = te[(~te.ser) & (te.pred != "P3")]
def snip(g, n=6): return [{"id": int(r.VAERS_ID), "true": r.label, "pred": r.pred, "text": r.SYMPTOM_TEXT[:260]}
                           for r in g.sample(min(n, len(g)), random_state=SEED).itertuples()]
errors = {"missed_serious": len(fn), "false_alarms": len(fp), "examples_missed": snip(fn), "examples_false_alarm": snip(fp, 4),
          "median_len_missed": int(fn.SYMPTOM_TEXT.str.len().median()), "median_len_all": int(te.SYMPTOM_TEXT.str.len().median())}

# ---------- 7. export ----------
os.makedirs(f"{a.out}/api/_lib", exist_ok=True); os.makedirs(f"{a.out}/public", exist_ok=True)
vocab = vec.vocabulary_; inv = [None] * len(vocab)
for k, v in vocab.items(): inv[v] = k
json.dump({"version": "baseline-tfidf-lr-v1", "classes": CLASSES, "terms": inv, "idf": [round(float(x), 4) for x in vec.idf_],
           "coef": [[round(float(x), 4) for x in row] for row in model.coef_], "intercept": [round(float(x), 4) for x in model.intercept_],
           "default_tau": default_tau, "rules": rules}, open(f"{a.out}/api/_lib/model.json", "w"), separators=(",", ":"))
json.dump({"n_reports": len(d), "split": {"train": len(tr), "val": len(va), "test": len(te)},
           "label_counts": d.label.value_counts().to_dict(), "test_label_counts": te.label.value_counts().to_dict(),
           "C": C, "ms_per_case": ms_per_case, "report": {k: rep[k] for k in CLASSES + ["macro avg"]}, "confusion": cm,
           "serious": serious, "model_plus_rules": combined, "by_age": grp("age_group"), "by_sex": grp("sex"),
           "curve": curve, "default_tau": default_tau, "aesi_validation": aesi_val, "errors": errors,
           "gemini": None}, open(f"{a.out}/public/metrics.json", "w"), indent=1)
# real, short test-set reports for the UI "load random real report" button (public VAERS text)
smp = pd.concat([g.sample(min(14, len(g)), random_state=SEED) for _, g in te[te.SYMPTOM_TEXT.str.len().between(80, 700)].groupby("label")])
json.dump([{"id": int(r.VAERS_ID), "age": None if pd.isna(r.AGE_YRS) else float(r.AGE_YRS), "sex": r.SEX, "reporter_label": r.label, "text": r.SYMPTOM_TEXT}
           for r in smp.itertuples()], open(f"{a.out}/public/samples.json", "w"))
te[["VAERS_ID", "label"]].assign(pred=pred, p1=Pte[:, 0], p2=Pte[:, 1], p3=Pte[:, 2]).to_csv(f"{a.out}/ml/test_predictions.csv", index=False)
print("exported.")
