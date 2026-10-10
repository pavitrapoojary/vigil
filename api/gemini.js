// Optional second-opinion model: Google Gemini with few-shot prompting. Key stays server-side.
const MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const SYSTEM = `You are a pharmacovigilance triage assistant. Read one vaccine adverse event report narrative and assign seriousness:
P1 = death or life-threatening event; P2 = hospitalisation or disability/incapacity; P3 = everything else (non-serious, including ER/clinic visits without admission).
Base the decision only on what the narrative states; do not assume. Flag adverse events of special interest (anaphylaxis, myocarditis/pericarditis, Guillain-Barre syndrome, thrombosis with thrombocytopenia, serious neurological events) if present and not negated.
Return JSON only: {"priority":"P1|P2|P3","confidence":0-1,"flagged_terms":[strings copied from the text],"special_interest":[strings],"rationale":"one or two sentences"}.
Examples:
Report: "62F, 2 days after dose, chest pain, troponin elevated, admitted for observation." -> {"priority":"P2","confidence":0.9,"flagged_terms":["chest pain","troponin","admitted"],"special_interest":["possible myocarditis"],"rationale":"Hospital admission with raised troponin suggests a serious cardiac event."}
Report: "Sore arm and mild fever for one day, resolved without treatment." -> {"priority":"P3","confidence":0.97,"flagged_terms":["sore arm","mild fever"],"special_interest":[],"rationale":"Mild, self-limiting reactions with no serious criteria."}
Report: "Patient found unresponsive 3 hours after vaccination, resuscitation unsuccessful, pronounced dead." -> {"priority":"P1","confidence":0.96,"flagged_terms":["unresponsive","resuscitation","dead"],"special_interest":[],"rationale":"Death reported after vaccination."}
Report: "Wrong vaccine given by pharmacy, no symptoms." -> {"priority":"P3","confidence":0.98,"flagged_terms":["wrong vaccine"],"special_interest":[],"rationale":"Medication error without any adverse outcome."}`;
const hits = new Map(); // best-effort per-instance rate limit
module.exports = async (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const key = process.env.GEMINI_API_KEY;
  if (!key) return res.status(503).json({ error: "GEMINI_API_KEY is not configured on the server." });
  const ip = req.headers["x-forwarded-for"] || "anon", now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < 60000);
  if (recent.length >= 10) return res.status(429).json({ error: "Rate limit: 10 requests per minute." });
  hits.set(ip, [...recent, now]);
  const text = (req.body || {}).text;
  if (typeof text !== "string" || text.trim().length < 10 || text.length > 8000) return res.status(400).json({ error: "Provide text between 10 and 8000 characters." });
  try {
    const t0 = Date.now();
    const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`, {
      method: "POST", headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({ systemInstruction: { parts: [{ text: SYSTEM }] }, contents: [{ role: "user", parts: [{ text: `Report: "${text}"` }] }],
        generationConfig: { temperature: 0, responseMimeType: "application/json" } }),
    });
    const j = await r.json();
    if (!r.ok) return res.status(502).json({ error: j.error?.message || "Gemini request failed" });
    const out = JSON.parse(j.candidates?.[0]?.content?.parts?.[0]?.text || "{}");
    if (!["P1", "P2", "P3"].includes(out.priority)) return res.status(502).json({ error: "Unexpected Gemini response" });
    res.status(200).json({ ...out, model: MODEL, latency_ms: Date.now() - t0 });
  } catch (e) { res.status(502).json({ error: "Gemini call failed: " + e.message }); }
};
