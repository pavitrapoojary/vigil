// Pure-JS inference for the exported TF-IDF + logistic-regression model (no Python needed at runtime).
const crypto = require("crypto");
const M = require("./model.json");
const IDX = new Map(M.terms.map((t, i) => [t, i]));
const ORDER = { P1: 0, P2: 1, P3: 2 };
const QUEUES = {
  P1: "Urgent medical review (15-day clock starts)",
  P2: "Medical reviewer queue",
  P3: "Non-serious batch queue",
  MANUAL: "Manual triage review",
};
const RULES = M.rules.aesi.map((r) => ({ ...r, re: new RegExp(r.pattern, "g") }));
const NEG = new Set(M.rules.negations);

function featurize(text) {
  const toks = (text.toLowerCase().match(/\b\w\w+\b/g) || []);
  const counts = new Map();
  const add = (g) => { const i = IDX.get(g); if (i !== undefined) counts.set(i, (counts.get(i) || 0) + 1); };
  for (let i = 0; i < toks.length; i++) { add(toks[i]); if (i + 1 < toks.length) add(toks[i] + " " + toks[i + 1]); }
  const feats = []; let norm = 0;
  for (const [i, c] of counts) { const v = (1 + Math.log(c)) * M.idf[i]; feats.push([i, v]); norm += v * v; }
  norm = Math.sqrt(norm) || 1;
  return feats.map(([i, v]) => [i, v / norm]);
}

function predict(text) {
  const f = featurize(text);
  const logits = M.coef.map((row, k) => f.reduce((s, [i, v]) => s + row[i] * v, M.intercept[k]));
  const mx = Math.max(...logits), ex = logits.map((l) => Math.exp(l - mx)), z = ex.reduce((a, b) => a + b, 0);
  const probs = ex.map((e) => e / z);
  // explanation: top positive contributions to the predicted class
  const k = probs.indexOf(Math.max(...probs));
  const contrib = f.map(([i, v]) => [M.terms[i], M.coef[k][i] * v]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 6);
  return { probs, k, drivers: contrib.map((c) => c[0]), nFeatures: f.length };
}

function runRules(text) {
  const t = text.toLowerCase(), hits = [];
  for (const r of RULES) {
    r.re.lastIndex = 0; let m;
    while ((m = r.re.exec(t))) {
      const before = (t.slice(Math.max(0, m.index - 40), m.index).match(/\w+/g) || []).slice(-4);
      if (!before.some((w) => NEG.has(w))) { hits.push({ id: r.id, label: r.label, min_priority: r.min_priority, term: m[0] }); break; }
    }
  }
  return hits;
}

function triage(text, { tau = M.default_tau, gemini = null } = {}) {
  const t0 = Date.now();
  const p = predict(text), hits = runRules(text);
  const classes = M.classes, modelPri = classes[p.k], conf = p.probs[p.k];
  let priority = modelPri; const reasons = [];
  reasons.push(`Model predicts ${modelPri} with confidence ${conf.toFixed(2)}.`);
  for (const h of hits) if (ORDER[h.min_priority] < ORDER[priority]) {
    priority = h.min_priority; reasons.push(`Rule override: "${h.term}" (${h.label}) raises priority to ${h.min_priority}.`);
  }
  let manual = false;
  if (hits.length) { manual = true; reasons.push("Special-interest rule fired, so a human medical reviewer must confirm."); }
  else if (conf < tau) { manual = true; reasons.push(`Confidence ${conf.toFixed(2)} is below the auto-route threshold ${tau.toFixed(2)}.`); }
  if (gemini && gemini.priority && ORDER[gemini.priority] < ORDER[priority]) {
    manual = true; reasons.push(`Second-opinion model rates this more serious (${gemini.priority}); escalated to manual review.`);
  }
  const queue = manual ? (priority === "P3" ? QUEUES.MANUAL : QUEUES[priority] + " (human-confirmed)") : QUEUES[priority];
  const decision = {
    priority, confidence: Number(conf.toFixed(3)), probabilities: Object.fromEntries(classes.map((c, i) => [c, Number(p.probs[i].toFixed(3))])),
    model_priority: modelPri, flagged_terms: [...new Set([...hits.map((h) => h.term), ...p.drivers])].slice(0, 8),
    special_interest: hits.map((h) => h.label), queue, routing: manual ? "MANUAL_REVIEW" : "AUTO_ROUTED",
    tau, reasons, model_version: M.version,
  };
  decision.audit = {
    case_hash: crypto.createHash("sha256").update(text).digest("hex").slice(0, 16),
    timestamp: new Date().toISOString(), latency_ms: Date.now() - t0, final_decision_by: "PENDING_HUMAN_CONFIRMATION",
  };
  return decision;
}
module.exports = { triage, QUEUES, DEFAULT_TAU: M.default_tau };
