// Zero-dependency local server: serves /public and runs the /api handlers exactly as Vercel would.
const http = require("http"), fs = require("fs"), path = require("path");
for (const f of [".env.local", ".env"]) if (fs.existsSync(f)) fs.readFileSync(f, "utf8").split("\n").forEach((l) => { const m = l.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2]; });
const types = { ".html": "text/html", ".json": "application/json", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
http.createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  if (url.pathname.startsWith("/api/")) {
    const file = path.join(__dirname, url.pathname + ".js"); let body = "";
    req.on("data", (c) => (body += c)).on("end", () => {
      try { req.body = body ? JSON.parse(body) : {}; } catch { req.body = {}; }
      res.status = (c) => { res.statusCode = c; return res; };
      res.json = (o) => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(o)); };
      if (!fs.existsSync(file) || url.pathname.includes("_lib")) return res.status(404).json({ error: "not found" });
      require(file)(req, res);
    }); return;
  }
  const f = path.join(__dirname, "public", url.pathname === "/" ? "index.html" : url.pathname);
  if (!f.startsWith(path.join(__dirname, "public")) || !fs.existsSync(f)) { res.statusCode = 404; return res.end("Not found"); }
  res.setHeader("Content-Type", types[path.extname(f)] || "application/octet-stream"); fs.createReadStream(f).pipe(res);
}).listen(process.env.PORT || 3000, () => console.log("Vigil running at http://localhost:" + (process.env.PORT || 3000)));
