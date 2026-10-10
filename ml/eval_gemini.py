"""Evaluate Gemini few-shot on a stratified sample of the held-out test set and write results into public/metrics.json.
Run:  GEMINI_API_KEY=... python ml/eval_gemini.py --data path/2025VAERSDATA.csv --n 150
Uses the same system prompt as api/gemini.js. Costs a few cents / fits the free tier for n<=150."""
import argparse, json, os, re, time, urllib.request
import pandas as pd, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=150); a = ap.parse_args()
key = os.environ["GEMINI_API_KEY"]; MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
src = open(os.path.join(os.path.dirname(__file__), "..", "api", "gemini.js")).read()
SYSTEM = re.search(r"const SYSTEM = `(.*?)`;", src, re.S).group(1)
pred = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_predictions.csv"))
d = pd.read_csv(a.data, encoding="latin-1", low_memory=False)[["VAERS_ID", "SYMPTOM_TEXT"]].drop_duplicates("VAERS_ID")
df = pred.merge(d, on="VAERS_ID"); df = df[df.SYMPTOM_TEXT.str.len().between(30, 4000)]
per = a.n // 3  # balanced so rare serious classes are represented
df = pd.concat([g.sample(min(per, len(g)), random_state=1) for _, g in df.groupby("label")])
rows = []
for r in df.itertuples():
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]}, "contents": [{"role": "user", "parts": [{"text": f'Report: "{r.SYMPTOM_TEXT}"'}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "x-goog-api-key": key})
    t0 = time.time()
    try:
        out = json.loads(json.loads(urllib.request.urlopen(req, timeout=60).read())["candidates"][0]["content"]["parts"][0]["text"])
        rows.append((r.label, r.pred, out["priority"], time.time() - t0))
    except Exception as e: print("skip", r.VAERS_ID, e)
    time.sleep(0.5)
g = pd.DataFrame(rows, columns=["true", "baseline", "gemini", "sec"]); ser = g.true.isin(["P1", "P2"])
def rec(col): return float((ser & g[col].isin(["P1", "P2"])).sum() / ser.sum())
def prec(col): return float((ser & g[col].isin(["P1", "P2"])).sum() / max(g[col].isin(["P1", "P2"]).sum(), 1))
res = {"model": MODEL, "n": len(g), "note": "Balanced sample (equal cases per class), so precision is NOT comparable to the full test set.",
       "gemini_serious_recall": rec("gemini"), "baseline_serious_recall_same_sample": rec("baseline"),
       "gemini_serious_precision": prec("gemini"), "baseline_serious_precision_same_sample": prec("baseline"),
       "gemini_accuracy": float((g.true == g.gemini).mean()), "baseline_accuracy": float((g.true == g.baseline).mean()),
       "gemini_sec_per_case": float(g.sec.mean())}
print(json.dumps(res, indent=1))
p = os.path.join(os.path.dirname(__file__), "..", "public", "metrics.json"); m = json.load(open(p)); m["gemini"] = res; json.dump(m, open(p, "w"), indent=1)
