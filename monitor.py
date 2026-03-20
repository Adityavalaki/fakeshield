"""
FakeShield — Auto News Monitor (3-Column Layout)
Fetches live news every 5 minutes, shows REAL | UNCERTAIN | FAKE in separate columns.

Run: python monitor.py
Open: http://localhost:5001
"""

import os, sys, time, sqlite3, logging, requests, threading, re
sys.path.insert(0, os.path.dirname(__file__))

from xml.etree import ElementTree as ET
from datetime  import datetime
from flask     import Flask, jsonify, request
from flask_socketio import SocketIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEEDS = {
    "The Hindu":       "https://www.thehindu.com/news/national/?service=rss",
    "Indian Express":  "https://indianexpress.com/feed/",
    "NDTV":            "https://feeds.feedburner.com/ndtvnews-india-news",
    "Times of India":  "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    "LiveMint":        "https://www.livemint.com/rss/news",
    "Hindustan Times": "https://www.hindustantimes.com/feeds/rss/india-news/rssfeed.xml",
    "BoomLive":        "https://www.boomlive.in/feed",
    "AltNews":         "https://www.altnews.in/feed/",
}

POLL_INTERVAL = 5

DB_PATH = "data/monitor.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT, text TEXT, url TEXT UNIQUE, source TEXT,
            label TEXT, confidence REAL, real_prob REAL, fake_prob REAL,
            verdict TEXT, detected_at TEXT)
    """)
    conn.commit(); conn.close()

def save_article(a):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""INSERT OR IGNORE INTO articles
            (title,text,url,source,label,confidence,real_prob,fake_prob,verdict,detected_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (a["title"],a["text"],a["url"],a["source"],a["label"],
             a["confidence"],a["real_prob"],a["fake_prob"],a["verdict"],a["detected_at"]))
        conn.commit()
    finally: conn.close()

def get_articles(limit=60, label=None, verdict=None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    q, p = "SELECT * FROM articles", []
    conds = []
    if label:   conds.append("label=?");   p.append(label)
    if verdict: conds.append("verdict=?"); p.append(verdict)
    if conds: q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY detected_at DESC LIMIT ?"; p.append(limit)
    rows = conn.execute(q, p).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    s = {
        "total":     conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
        "fake":      conn.execute("SELECT COUNT(*) FROM articles WHERE label='FAKE'").fetchone()[0],
        "real":      conn.execute("SELECT COUNT(*) FROM articles WHERE label='REAL'").fetchone()[0],
        "uncertain": conn.execute("SELECT COUNT(*) FROM articles WHERE verdict='Uncertain'").fetchone()[0],
    }
    conn.close(); return s

def get_seen_urls():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT url FROM articles").fetchall()
    conn.close(); return {r[0] for r in rows}


class Predictor:
    def __init__(self):
        self.model = self.tokenizer = self.device = None
        self._load()

    def _load(self):
        ckpt = os.path.join("models","best_checkpoint.pt")
        tok  = os.path.join("models","tokenizer")
        if not os.path.exists(ckpt):
            log.warning("No trained model — using keyword demo mode"); return
        try:
            import torch
            from transformers import AutoTokenizer
            from src.model import FakeNewsClassifier
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(tok)
            self.model = FakeNewsClassifier()
            state = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.model.to(self.device); self.model.eval()
            log.info(f"Model loaded on {self.device}")
        except Exception as e: log.error(f"Model load failed: {e}")

    def predict(self, text):
        if self.model is not None:
            try:
                import torch, torch.nn.functional as F, config
                enc = self.tokenizer(text, max_length=config.MAX_SEQ_LENGTH,
                    padding="max_length", truncation=True, return_tensors="pt")
                with torch.no_grad():
                    probs = F.softmax(self.model(
                        enc["input_ids"].to(self.device),
                        enc["attention_mask"].to(self.device))["logits"],dim=-1).cpu().numpy()[0]
                pred = int(probs.argmax()); conf = float(probs[pred])
                label = "FAKE" if pred==1 else "REAL"
                verdict = ("Fake News" if label=="FAKE" else "Real News") if conf>=config.CONFIDENCE_THRESHOLD else "Uncertain"
                return {"label":label,"confidence":round(conf,4),"verdict":verdict,
                        "real_prob":round(float(probs[0]),4),"fake_prob":round(float(probs[1]),4)}
            except Exception as e: log.error(f"Predict error: {e}")

        import random, hashlib
        fake_kw = ["shocking","breaking","exposed","secret","deep state","hoax","wake up","miracle"]
        real_kw = ["according to","study","research","official","confirmed","peer-reviewed","published"]
        tl = text.lower()
        seed = int(hashlib.md5(text.encode()).hexdigest(),16) % 1000
        rng  = random.Random(seed)
        base = max(0.1,min(0.9, 0.42 + sum(k in tl for k in fake_kw)*0.1
                                      - sum(k in tl for k in real_kw)*0.1
                                      + rng.uniform(-0.08,0.08)))
        label = "FAKE" if base>0.5 else "REAL"
        conf  = base if label=="FAKE" else 1-base
        verdict = ("Fake News" if label=="FAKE" else "Real News") if conf>=0.75 else "Uncertain"
        return {"label":label,"confidence":round(conf,4),"verdict":verdict,
                "real_prob":round(1-base,4),"fake_prob":round(base,4),"demo":True}


def fetch_feed(source, url):
    try:
        res  = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        root = ET.fromstring(res.content)
        arts = []
        for item in root.iter("item"):
            title = item.find("title"); link = item.find("link"); desc = item.find("description")
            if title is None or link is None: continue
            t = (title.text or "").strip()
            d = re.sub('<[^<]+?>','',(desc.text or "") if desc is not None else "").strip()
            l = (link.text or "").strip()
            if not l: continue
            arts.append({"title":t,"text":t+" "+d,"url":l,"source":source})
        return arts[:15]
    except Exception as e:
        log.warning(f"Feed error [{source}]: {e}"); return []


predictor = seen_urls = socketio_ref = None

def poll():
    global seen_urls
    log.info(f"[{datetime.now().strftime('%H:%M:%S')}] Polling {len(FEEDS)} feeds...")
    new_count = 0
    for source, url in FEEDS.items():
        for art in fetch_feed(source, url):
            if not art["url"] or art["url"] in seen_urls: continue
            result = predictor.predict(art["text"])
            record = {**art,"label":result["label"],"confidence":result["confidence"],
                      "real_prob":result["real_prob"],"fake_prob":result["fake_prob"],
                      "verdict":result["verdict"],"detected_at":datetime.now().isoformat()}
            save_article(record); seen_urls.add(art["url"]); new_count += 1
            if socketio_ref: socketio_ref.emit("new_article", record)
            log.info(f"  [{result['label']:4s} {result['confidence']*100:.0f}%] {source}: {art['title'][:55]}...")
            time.sleep(0.1)
    log.info(f"Poll done — {new_count} new articles")

def monitor_loop():
    while True:
        try: poll()
        except Exception as e: log.error(f"Poll error: {e}")
        time.sleep(POLL_INTERVAL * 60)


app      = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>FakeShield — Live Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
<style>
  :root{--bg:#0a0b0d;--surface:#111318;--border:#1e2130;--accent:#e8ff47;
        --red:#ff4444;--green:#44ff99;--yellow:#ffaa44;--text:#e2e8f0;--muted:#4a5160;--sub:#7a8599}
  *{box-sizing:border-box;margin:0;padding:0}
  html,body{height:100%}
  body{font-family:'Syne',sans-serif;background:var(--bg);color:var(--text);
       display:flex;flex-direction:column;height:100vh;overflow:hidden}
  body::before{content:'';position:fixed;inset:0;
    background-image:linear-gradient(rgba(232,255,71,.03) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(232,255,71,.03) 1px,transparent 1px);
    background-size:48px 48px;pointer-events:none;z-index:0}

  header{z-index:10;padding:16px 32px;border-bottom:1px solid var(--border);
         display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
         position:relative}
  .logo{font-size:20px;font-weight:800}.logo span{color:var(--accent)}
  .logo small{font-size:12px;color:var(--muted);font-weight:400;margin-left:10px}
  .live-badge{display:flex;align-items:center;gap:8px;font-family:'IBM Plex Mono',monospace;
              font-size:11px;color:var(--green);border:1px solid rgba(68,255,153,.3);
              padding:6px 14px;border-radius:20px}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--green);animation:pulse 1.5s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

  .stats-row{z-index:10;display:grid;grid-template-columns:repeat(4,1fr);
             gap:1px;background:var(--border);border-bottom:1px solid var(--border);
             flex-shrink:0;position:relative}
  .stat-box{background:var(--surface);padding:16px 28px}
  .stat-num{font-family:'IBM Plex Mono',monospace;font-size:26px;font-weight:500}
  .stat-num.t{color:var(--accent)}.stat-num.f{color:var(--red)}
  .stat-num.r{color:var(--green)}.stat-num.u{color:var(--yellow)}
  .stat-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:2px}

  .controls{z-index:10;padding:10px 32px;border-bottom:1px solid var(--border);
            display:flex;align-items:center;gap:12px;flex-shrink:0;position:relative}
  .next-poll{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted)}
  .demo-tag{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--yellow);
            background:rgba(255,170,68,.08);border:1px solid rgba(255,170,68,.2);
            padding:4px 10px;border-radius:10px;display:none}
  .poll-btn{margin-left:auto;background:var(--accent);color:#000;border:none;
            font-family:'Syne',sans-serif;font-weight:700;font-size:12px;
            padding:7px 18px;border-radius:20px;cursor:pointer;transition:transform .2s}
  .poll-btn:hover{transform:translateY(-1px)}

  /* 3 columns */
  .columns{display:grid;grid-template-columns:1fr 1fr 1fr;
           gap:1px;background:var(--border);flex:1;min-height:0;position:relative;z-index:10}

  .col{background:var(--bg);display:flex;flex-direction:column;overflow:hidden}

  .col-header{padding:12px 16px;display:flex;align-items:center;
              justify-content:space-between;flex-shrink:0;border-bottom:1px solid var(--border)}
  .col-real   .col-header{border-top:3px solid var(--green)}
  .col-fake   .col-header{border-top:3px solid var(--red)}
  .col-uncertain .col-header{border-top:3px solid var(--yellow)}

  .col-title{font-size:13px;font-weight:700;display:flex;align-items:center;gap:7px}
  .col-real    .col-title{color:var(--green)}
  .col-fake    .col-title{color:var(--red)}
  .col-uncertain .col-title{color:var(--yellow)}

  .col-count{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:3px 9px;border-radius:10px}
  .col-real    .col-count{background:rgba(68,255,153,.1);color:var(--green)}
  .col-fake    .col-count{background:rgba(255,68,68,.1);color:var(--red)}
  .col-uncertain .col-count{background:rgba(255,170,68,.1);color:var(--yellow)}

  .col-feed{flex:1;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:7px}
  .col-feed::-webkit-scrollbar{width:3px}
  .col-feed::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

  .acard{background:var(--surface);border:1px solid var(--border);border-radius:10px;
         padding:11px 13px;text-decoration:none;color:inherit;display:block;
         transition:border-color .2s;animation:fadeIn .3s ease}
  .acard:hover{border-color:rgba(232,255,71,.4)}
  .acard.flash{animation:flash .5s ease,fadeIn .3s ease}
  @keyframes fadeIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
  @keyframes flash{0%{box-shadow:0 0 0 2px rgba(232,255,71,.5)}100%{box-shadow:none}}

  .acard-title{font-size:12px;font-weight:600;line-height:1.45;margin-bottom:7px;
               display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
  .acard-meta{display:flex;align-items:center;justify-content:space-between;gap:6px}
  .acard-source{font-family:'IBM Plex Mono',monospace;font-size:10px;color:var(--muted);
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .acard-conf{font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;flex-shrink:0}
  .col-real    .acard-conf{color:var(--green)}
  .col-fake    .acard-conf{color:var(--red)}
  .col-uncertain .acard-conf{color:var(--yellow)}

  .empty{text-align:center;color:var(--muted);padding:32px 16px;
         font-family:'IBM Plex Mono',monospace;font-size:11px;line-height:1.8}
</style>
</head>
<body>

<header>
  <div class="logo">Fake<span>Shield</span><small>Live Monitor</small></div>
  <div class="live-badge"><div class="dot"></div>Auto-polling every 5 min</div>
</header>

<div class="stats-row">
  <div class="stat-box"><div class="stat-num t" id="sTotal">0</div><div class="stat-label">Total Analyzed</div></div>
  <div class="stat-box"><div class="stat-num f" id="sFake">0</div><div class="stat-label">Flagged Fake</div></div>
  <div class="stat-box"><div class="stat-num r" id="sReal">0</div><div class="stat-label">Verified Real</div></div>
  <div class="stat-box"><div class="stat-num u" id="sUncertain">0</div><div class="stat-label">Uncertain</div></div>
</div>

<div class="controls">
  <span class="next-poll" id="nextPoll">Next poll in 5:00</span>
  <span class="demo-tag" id="demoTag">⚠ Demo Mode — run python train.py for AI predictions</span>
  <button class="poll-btn" onclick="pollNow()">⚡ Poll Now</button>
</div>

<div class="columns">

  <div class="col col-real">
    <div class="col-header">
      <div class="col-title">✅ Real News</div>
      <span class="col-count" id="countReal">0</span>
    </div>
    <div class="col-feed" id="feedReal">
      <div class="empty">⏳ Fetching real news...</div>
    </div>
  </div>

  <div class="col col-uncertain">
    <div class="col-header">
      <div class="col-title">⚠️ Uncertain</div>
      <span class="col-count" id="countUncertain">0</span>
    </div>
    <div class="col-feed" id="feedUncertain">
      <div class="empty">⏳ Fetching articles...</div>
    </div>
  </div>

  <div class="col col-fake">
    <div class="col-header">
      <div class="col-title">🚨 Fake News</div>
      <span class="col-count" id="countFake">0</span>
    </div>
    <div class="col-feed" id="feedFake">
      <div class="empty">⏳ Fetching flagged articles...</div>
    </div>
  </div>

</div>

<script>
  const socket = io();
  let countdown = 300;

  setInterval(() => {
    countdown--;
    if (countdown <= 0) countdown = 300;
    const m = Math.floor(countdown/60), s = countdown%60;
    document.getElementById('nextPoll').textContent =
      `Next poll in ${m}:${s.toString().padStart(2,'0')}`;
  }, 1000);

  function cardHTML(a, flash) {
    const time = a.detected_at ? new Date(a.detected_at).toLocaleTimeString() : '';
    return `<a class="acard ${flash?'flash':''}" href="${a.url||'#'}" target="_blank">
      <div class="acard-title">${a.title||'No title'}</div>
      <div class="acard-meta">
        <span class="acard-source">${a.source} · ${time}</span>
        <span class="acard-conf">${Math.round(a.confidence*100)}%</span>
      </div>
    </a>`;
  }

  function renderCol(feedId, countId, articles) {
    document.getElementById(feedId).innerHTML = articles.length
      ? articles.map(a => cardHTML(a,false)).join('')
      : '<div class="empty">No articles yet<br>Waiting for next poll...</div>';
    document.getElementById(countId).textContent = articles.length;
  }

  async function loadAll() {
    const [rRes,fRes,uRes] = await Promise.all([
      fetch('/api/articles?label=REAL&limit=50'),
      fetch('/api/articles?label=FAKE&limit=50'),
      fetch('/api/articles?verdict=Uncertain&limit=50'),
    ]);
    const [rData,fData,uData] = await Promise.all([rRes.json(),fRes.json(),uRes.json()]);
    renderCol('feedReal',      'countReal',      rData.filter(a=>a.verdict!=='Uncertain'));
    renderCol('feedFake',      'countFake',      fData.filter(a=>a.verdict!=='Uncertain'));
    renderCol('feedUncertain', 'countUncertain', uData);
  }

  async function loadStats() {
    const d = await (await fetch('/api/stats')).json();
    document.getElementById('sTotal').textContent    = d.total;
    document.getElementById('sFake').textContent     = d.fake;
    document.getElementById('sReal').textContent     = d.real;
    document.getElementById('sUncertain').textContent = d.uncertain;
  }

  socket.on('new_article', (a) => {
    countdown = 300;
    let feedId, countId;
    if      (a.verdict==='Uncertain') { feedId='feedUncertain'; countId='countUncertain'; }
    else if (a.label==='REAL')        { feedId='feedReal';      countId='countReal'; }
    else                              { feedId='feedFake';      countId='countFake'; }

    const feed = document.getElementById(feedId);
    const empty = feed.querySelector('.empty');
    if (empty) empty.remove();
    feed.insertAdjacentHTML('afterbegin', cardHTML(a, true));

    const ce = document.getElementById(countId);
    ce.textContent = parseInt(ce.textContent||0)+1;
    loadStats();
    if (a.demo) document.getElementById('demoTag').style.display='inline-block';
  });

  async function pollNow() {
    const btn = document.querySelector('.poll-btn');
    btn.textContent='⏳ Polling...'; btn.disabled=true; countdown=300;
    await fetch('/api/poll');
    await Promise.all([loadAll(), loadStats()]);
    btn.textContent='⚡ Poll Now'; btn.disabled=false;
  }

  loadAll(); loadStats();
  setInterval(loadStats, 15000);
</script>
</body>
</html>"""


@app.route("/")
def dashboard(): return DASHBOARD_HTML

@app.route("/api/articles")
def api_articles():
    return jsonify(get_articles(
        limit=int(request.args.get("limit",60)),
        label=request.args.get("label"),
        verdict=request.args.get("verdict")))

@app.route("/api/stats")
def api_stats(): return jsonify(get_stats())

@app.route("/api/poll")
def api_poll():
    threading.Thread(target=poll, daemon=True).start()
    return jsonify({"status":"polling"})


if __name__ == "__main__":
    init_db()
    seen_urls    = get_seen_urls()
    predictor    = Predictor()
    socketio_ref = socketio

    threading.Thread(target=monitor_loop, daemon=True).start()

    print("\n" + "="*55)
    print("  FakeShield Live Monitor — 3 Column View")
    print("  Open : http://localhost:5001")
    print("  Cols : Real News | Uncertain | Fake News")
    print("  Polls: every 5 minutes automatically")
    print("  Press Ctrl+C to stop")
    print("="*55 + "\n")

    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
