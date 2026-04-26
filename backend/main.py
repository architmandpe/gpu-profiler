"""
FastAPI backend — GPU Inference Profiler v2

Routes:
  GET  /                       → frontend
  GET  /report/{run_id}        → shareable report page
  GET  /health
  GET  /api/models
  GET  /api/gpus
  POST /api/profile            → run analysis (benchmark-calibrated)
  POST /api/lead               → capture email
  GET  /api/admin/leads        → export leads (requires ADMIN_SECRET header)
  GET  /api/admin/stats        → run stats (requires ADMIN_SECRET header)
"""

from __future__ import annotations
import os
import time
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from gpu_specs import get_gpu_list
from model_registry import get_model_ids, get_model_list
from profiler import run_profile_v2, ProfileReport
from leads import save_run, save_lead, get_all_leads, get_run_stats, get_run_by_id, DB_PATH
import sqlite3

app = FastAPI(title="GPU Inference Profiler", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "changeme")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ProfileRequest(BaseModel):
    model_name: str
    gpu_name: str
    batch_size: int = Field(default=1, ge=1, le=512)
    context_length: int = Field(default=2048, ge=64, le=131072)
    precision: str = Field(default="fp16")
    has_flash_attn: bool = False
    has_continuous_batching: bool = False
    serving_framework: str = ""
    gpu_count: int = Field(default=4, ge=1, le=512)

class LeadRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=200)
    company: str = Field(default="", max_length=200)
    run_id: str = Field(default="", max_length=50)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    p = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=p.read_text() if p.exists() else "<h1>frontend not found</h1>")


@app.get("/report/{run_id}", response_class=HTMLResponse)
async def serve_report(run_id: str):
    """Shareable report page — fetches the stored run and renders it."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT raw_json FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")

    report_data = json.loads(row["raw_json"])
    html = _render_share_page(run_id, report_data)
    return HTMLResponse(content=html)


@app.get("/api/report/{run_id}")
async def get_report_json(run_id: str):
    """Return stored report as JSON."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT raw_json FROM runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    return JSONResponse(content=json.loads(row["raw_json"]))


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/models")
async def list_models():
    return {"models": get_model_ids(), "display_names": get_model_list()}


@app.get("/api/gpus")
async def list_gpus():
    return {"gpus": get_gpu_list()}


@app.post("/api/profile")
async def profile(req: ProfileRequest):
    result = run_profile_v2(
        model_name=req.model_name,
        gpu_name=req.gpu_name,
        batch_size=req.batch_size,
        context_length=req.context_length,
        precision=req.precision,
        has_flash_attn=req.has_flash_attn,
        has_continuous_batching=req.has_continuous_batching,
        serving_framework=req.serving_framework,
        gpu_count=req.gpu_count,
    )

    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    report_dict = _report_to_dict(result)

    try:
        run_id = save_run(report_dict)
        report_dict["run_id"] = run_id
    except Exception:
        report_dict["run_id"] = ""

    return JSONResponse(content=report_dict)


@app.post("/api/lead")
async def capture_lead(req: LeadRequest):
    if not req.email or "@" not in req.email:
        raise HTTPException(status_code=422, detail="Invalid email address.")
    try:
        lead_id = save_lead(email=req.email, run_id=req.run_id or None, company=req.company or None)
        return {"ok": True, "lead_id": lead_id}
    except Exception:
        raise HTTPException(status_code=500, detail="Could not save lead.")



@app.get("/report/{run_id}", response_class=HTMLResponse)
async def shareable_report(run_id: str):
    """Serve a pre-rendered shareable report page with embedded JSON data."""
    report_data = get_run_by_id(run_id)
    if not report_data:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    template_path = FRONTEND_DIR / "report.html"
    if not template_path.exists():
        raise HTTPException(status_code=500, detail="Report template not found.")

    # Inject the report JSON into the HTML template
    import json as _json
    html = template_path.read_text()
    # Replace the placeholder with actual data
    json_str = _json.dumps(report_data)
    html = html.replace('__REPORT_JSON__', json_str)
    return HTMLResponse(content=html, status_code=200)

@app.get("/api/admin/leads")
async def admin_leads(x_admin_secret: str = Header(default="")):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    return {"leads": get_all_leads()}


@app.get("/api/admin/stats")
async def admin_stats(x_admin_secret: str = Header(default="")):
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized.")
    return get_run_stats()


# ── Serialization ─────────────────────────────────────────────────────────────

def _report_to_dict(report: ProfileReport) -> dict:
    d = {
        "model_id": report.model_id,
        "model_display": report.model_display,
        "gpu_name": report.gpu_name,
        "gpu_display": report.gpu_display,
        "profiling_mode": report.profiling_mode,
        "efficiency_score": report.efficiency_score,
        "headline": report.headline,
        "subheadline": report.subheadline,
        "profiled_at": report.profiled_at,
        "throughput": {
            "measured_tokens_per_sec": report.throughput.measured_tokens_per_sec,
            "theoretical_peak_tokens_per_sec": report.throughput.theoretical_peak_tokens_per_sec,
            "efficiency_pct": report.throughput.efficiency_pct,
            "batch_size": report.throughput.batch_size,
            "context_length": report.throughput.context_length,
            "precision": report.throughput.precision,
        },
        "memory": {
            "model_weights_gb": report.memory.model_weights_gb,
            "kv_cache_gb": report.memory.kv_cache_gb,
            "total_used_gb": report.memory.total_used_gb,
            "available_gb": report.memory.available_gb,
            "utilization_pct": report.memory.utilization_pct,
            "bandwidth_utilization_pct": report.memory.bandwidth_utilization_pct,
            "bound": report.memory.bound,
        },
        "cost": {
            "monthly_gpu_cost_usd": report.cost.monthly_gpu_cost_usd,
            "optimized_monthly_cost_usd": report.cost.optimized_monthly_cost_usd,
            "monthly_savings_usd": report.cost.monthly_savings_usd,
            "savings_pct": report.cost.savings_pct,
            "assumption_basis": report.cost.assumption_basis,
        },
        "bottlenecks": [
            {
                "rank": b.rank,
                "category": b.category,
                "severity": b.severity,
                "title": b.title,
                "description": b.description,
                "estimated_speedup": b.estimated_speedup,
                "fix": b.fix,
            }
            for b in report.bottlenecks
        ],
        "optimized_tps": getattr(report, "_optimized_tps", None),
        "bench_note": getattr(report, "_bench_note", ""),
    }
    return d


# ── Shareable report HTML ─────────────────────────────────────────────────────

def _render_share_page(run_id: str, d: dict) -> str:
    efficiency = d["efficiency_score"]
    gap = round(100 - efficiency, 0)
    savings = d["cost"]["monthly_savings_usd"]
    model = d["model_display"]
    gpu = d["gpu_display"]

    eff_color = "#4ade80" if efficiency >= 70 else "#fbbf24" if efficiency >= 45 else "#f87171"

    bottleneck_rows = ""
    for b in d["bottlenecks"][:3]:
        sev_color = {"critical": "#f87171", "high": "#fbbf24", "medium": "#60a5fa", "low": "#6b6b7a"}.get(b["severity"], "#6b6b7a")
        bottleneck_rows += f"""
        <div style="border:1px solid #242429;border-radius:8px;padding:14px 16px;margin-bottom:8px;background:#111114">
          <div style="font-size:13px;font-weight:500;color:#e8e8ec;margin-bottom:4px">
            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:{sev_color};margin-right:7px;vertical-align:middle"></span>
            {b['title']}
          </div>
          <div style="font-size:12px;color:#6b6b7a;margin-bottom:6px">{b['description'][:140]}...</div>
          <div style="font-family:monospace;font-size:11px;color:#4ade80;background:rgba(74,222,128,.06);border:1px solid rgba(74,222,128,.15);padding:5px 8px;border-radius:4px">→ {b['fix'][:100]}</div>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model} on {gpu} — GPU Inference Efficiency Report</title>
<meta property="og:title" content="{model} inference is {gap:.0f}% below peak efficiency">
<meta property="og:description" content="Running on {gpu}. Estimated ${savings:,.0f}/month in avoidable GPU spend. See the full analysis.">
<meta name="twitter:card" content="summary_large_image">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0a0a0b;color:#e8e8ec;font-family:'IBM Plex Sans',sans-serif;font-size:14px;min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding:40px 20px 80px}}
.card{{background:#111114;border:1px solid #242429;border-radius:12px;padding:28px;width:100%;max-width:640px;margin-bottom:16px}}
.tag{{font-family:monospace;font-size:11px;color:#4ade80;background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.2);padding:3px 8px;border-radius:4px;display:inline-block;margin-bottom:14px}}
h1{{font-size:22px;font-weight:600;line-height:1.3;margin-bottom:6px}}
.sub{{color:#6b6b7a;font-size:13px;margin-bottom:20px}}
.score-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.score-label{{font-size:11px;color:#6b6b7a;font-family:monospace;width:110px;flex-shrink:0}}
.score-bar{{flex:1;height:6px;background:#18181d;border-radius:3px;overflow:hidden}}
.score-fill{{height:100%;border-radius:3px}}
.score-val{{font-family:monospace;font-size:12px;font-weight:500;width:40px;text-align:right}}
.metrics{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:18px 0}}
.metric{{background:#18181d;border-radius:6px;padding:12px}}
.mlabel{{font-size:10px;color:#6b6b7a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.mval{{font-family:monospace;font-size:18px;font-weight:500}}
.msub{{font-size:11px;color:#6b6b7a;margin-top:2px}}
.section-label{{font-size:11px;font-weight:500;color:#6b6b7a;text-transform:uppercase;letter-spacing:.07em;margin:20px 0 10px;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:#242429}}
.cost-grid{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;background:rgba(74,222,128,.04);border:1px solid rgba(74,222,128,.2);border-radius:8px;padding:16px}}
.clabel{{font-size:10px;color:#6b6b7a;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}}
.cval{{font-family:monospace;font-size:17px;font-weight:500}}
.savings .cval{{color:#4ade80}}
.cta{{background:#111114;border:1px solid #242429;border-radius:12px;padding:24px;width:100%;max-width:640px;text-align:center}}
.cta-title{{font-size:16px;font-weight:600;margin-bottom:8px}}
.cta-sub{{font-size:13px;color:#6b6b7a;margin-bottom:16px}}
.cta-btn{{display:inline-block;background:#4ade80;color:#0a0a0b;border-radius:7px;font-family:monospace;font-size:13px;font-weight:500;padding:10px 24px;text-decoration:none;cursor:pointer}}
.footer{{font-size:12px;color:#6b6b7a;margin-top:24px;text-align:center}}
@media(max-width:500px){{.metrics,.cost-grid{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<div class="card">
  <div class="tag">{model} · {gpu}</div>
  <h1>{d['headline']}</h1>
  <div class="sub">{d['subheadline']}</div>

  <div class="score-row">
    <span class="score-label">Efficiency</span>
    <div class="score-bar"><div class="score-fill" style="width:{efficiency}%;background:{eff_color}"></div></div>
    <span class="score-val" style="color:{eff_color}">{efficiency:.0f}%</span>
  </div>
  <div class="score-row">
    <span class="score-label">BW utilized</span>
    <div class="score-bar"><div class="score-fill" style="width:{d['memory']['bandwidth_utilization_pct']}%;background:#60a5fa"></div></div>
    <span class="score-val">{d['memory']['bandwidth_utilization_pct']:.0f}%</span>
  </div>

  <div class="metrics">
    <div class="metric">
      <div class="mlabel">Measured</div>
      <div class="mval">{int(d['throughput']['measured_tokens_per_sec']):,}</div>
      <div class="msub">tokens/sec</div>
    </div>
    <div class="metric">
      <div class="mlabel">Peak possible</div>
      <div class="mval">{int(d['throughput']['theoretical_peak_tokens_per_sec']):,}</div>
      <div class="msub">tokens/sec</div>
    </div>
    <div class="metric">
      <div class="mlabel">Memory bound</div>
      <div class="mval" style="font-size:14px">{d['memory']['bound'].replace('-', ' ').title()}</div>
      <div class="msub">{d['memory']['model_weights_gb']:.1f}GB weights</div>
    </div>
  </div>

  <div class="section-label">Top bottlenecks</div>
  {bottleneck_rows}

  <div class="section-label">Monthly cost impact</div>
  <div class="cost-grid">
    <div>
      <div class="clabel">Current cost</div>
      <div class="cval">${int(d['cost']['monthly_gpu_cost_usd']):,}</div>
      <div style="font-size:11px;color:#6b6b7a">{d['cost']['assumption_basis'][:40]}</div>
    </div>
    <div>
      <div class="clabel">Optimized</div>
      <div class="cval">${int(d['cost']['optimized_monthly_cost_usd']):,}</div>
      <div style="font-size:11px;color:#6b6b7a">after fixes applied</div>
    </div>
    <div class="savings">
      <div class="clabel">Monthly savings</div>
      <div class="cval">${int(d['cost']['monthly_savings_usd']):,}</div>
      <div style="font-size:11px;color:#4ade80">{d['cost']['savings_pct']:.0f}% reduction</div>
    </div>
  </div>
</div>

<div class="cta">
  <div class="cta-title">Analyze your own inference stack</div>
  <div class="cta-sub">Free. No signup. Enter your model + GPU, get your efficiency report in 10 seconds.</div>
  <a class="cta-btn" href="/">Run your own analysis →</a>
</div>

<div class="footer">
  inference.profile · Report ID: {run_id} · Based on roofline analysis + community benchmarks
</div>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
