#!/usr/bin/env python3
import sys, os, csv
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from leads import get_all_leads, get_run_stats

def show_stats():
    s = get_run_stats()
    print(f"""
  Total analyses run : {s['total_runs']}
  Emails captured    : {s['total_leads']}
  Avg efficiency     : {s['avg_efficiency_pct']}%
  Avg monthly savings: ${s['avg_monthly_savings_usd']:,.0f}
""")
    for m in s['top_models']:
        print(f"    {m['count']:>4}x  {m['model']}")

def show_leads(export_csv=False):
    leads = get_all_leads()
    if not leads:
        print("No leads yet. Share your tool!"); return
    if export_csv:
        with open("leads_export.csv","w",newline="") as f:
            w = csv.DictWriter(f, fieldnames=leads[0].keys())
            w.writeheader(); w.writerows(leads)
        print(f"Exported {len(leads)} leads to leads_export.csv"); return
    print(f"\n{'TIME':<20} {'EMAIL':<35} {'MODEL':<25} {'SAVINGS/MO'}")
    print("─"*90)
    for l in leads:
        ts      = datetime.fromtimestamp(l['ts']).strftime('%Y-%m-%d %H:%M')
        email   = (l.get('email') or '')[:34]
        model   = (l.get('model') or 'unknown')[:24]
        savings = f"${l.get('savings_usd') or 0:,.0f}" if l.get('savings_usd') else '—'
        print(f"{ts:<20} {email:<35} {model:<25} {savings}")
    print(f"\n{len(leads)} total leads\n")

def show_followup():
    leads = get_all_leads()
    if not leads: print("No leads yet."); return
    for l in leads[:5]:
        model    = l.get('model','your model')
        savings  = l.get('savings_usd', 0)
        eff      = l.get('efficiency', 0)
        gap      = round(100 - eff, 0) if eff else 40
        print(f"TO: {l.get('email','')}")
        print(f"SUBJECT: Your {model} stack — {gap:.0f}% efficiency gap we can fix\n")
        print(f"""Hi,

You ran our profiler on {model} last week. We found your stack
is {gap:.0f}% below achievable efficiency — roughly ${savings:,.0f}/month avoidable.

We fix this in a 2-week engagement for $2,000. Full refund if we
don't improve efficiency by at least 20%.

Want a 20-minute call this week?\n""")
        print("─"*60+"\n")

if __name__=="__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "stats"
    if cmd=="stats": show_stats()
    elif cmd=="leads": show_leads("csv" in sys.argv)
    elif cmd=="follow-up": show_followup()
    else: print("Usage: python admin.py [stats|leads|leads csv|follow-up]")
