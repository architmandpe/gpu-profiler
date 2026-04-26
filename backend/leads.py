import sqlite3, json, time, hashlib
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "leads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, ts REAL NOT NULL, model TEXT NOT NULL,
        gpu TEXT NOT NULL, efficiency REAL NOT NULL, savings_usd REAL NOT NULL,
        bottleneck1 TEXT, bottleneck2 TEXT, bottleneck3 TEXT, raw_json TEXT NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,
        email TEXT NOT NULL, company TEXT, run_id TEXT REFERENCES runs(id), notes TEXT)""")
    conn.commit(); conn.close()

def generate_run_id(model, gpu, ts):
    return hashlib.sha1(f"{model}:{gpu}:{ts:.3f}".encode()).hexdigest()[:12]

def save_run(report_dict):
    ts = time.time()
    run_id = generate_run_id(report_dict["model_id"], report_dict["gpu_name"], ts)
    bns = report_dict.get("bottlenecks", [])
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""INSERT OR IGNORE INTO runs
            (id,ts,model,gpu,efficiency,savings_usd,bottleneck1,bottleneck2,bottleneck3,raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            run_id, ts,
            report_dict["model_display"], report_dict["gpu_display"],
            report_dict["efficiency_score"], report_dict["cost"]["monthly_savings_usd"],
            bns[0]["title"] if len(bns)>0 else None,
            bns[1]["title"] if len(bns)>1 else None,
            bns[2]["title"] if len(bns)>2 else None,
            json.dumps(report_dict)))
        conn.commit()
    finally:
        conn.close()
    return run_id

def save_lead(email, run_id=None, company=None):
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute("INSERT INTO leads (ts,email,company,run_id) VALUES (?,?,?,?)",
                           (time.time(), email.strip().lower(), company, run_id))
        conn.commit(); return cur.lastrowid
    finally:
        conn.close()

def get_run_by_id(run_id):
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT raw_json FROM runs WHERE id=?", (run_id,)).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        conn.close()

def get_all_leads():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""SELECT l.ts,l.email,l.company,r.model,r.gpu,
            r.efficiency,r.savings_usd,r.bottleneck1,r.bottleneck2
            FROM leads l LEFT JOIN runs r ON r.id=l.run_id ORDER BY l.ts DESC""").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def get_run_stats():
    conn = sqlite3.connect(DB_PATH)
    try:
        total_runs  = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        avg_eff     = conn.execute("SELECT AVG(efficiency) FROM runs").fetchone()[0] or 0
        avg_save    = conn.execute("SELECT AVG(savings_usd) FROM runs").fetchone()[0] or 0
        top_models  = conn.execute("""SELECT model, COUNT(*) as cnt FROM runs
            GROUP BY model ORDER BY cnt DESC LIMIT 5""").fetchall()
        return {"total_runs": total_runs, "total_leads": total_leads,
                "avg_efficiency_pct": round(avg_eff,1),
                "avg_monthly_savings_usd": round(avg_save,0),
                "top_models": [{"model": r[0], "count": r[1]} for r in top_models]}
    finally:
        conn.close()

init_db()
