const { triage } = require("./_lib/triage");
module.exports = (req, res) => {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });
  const { text, tau, gemini } = req.body || {};
  if (typeof text !== "string" || text.trim().length < 10) return res.status(400).json({ error: "Provide a report narrative (min 10 characters)." });
  if (text.length > 20000) return res.status(413).json({ error: "Narrative too long (max 20,000 characters)." });
  const t = Number(tau); res.status(200).json(triage(text, { tau: t >= 0.3 && t <= 0.99 ? t : undefined, gemini }));
};
