# Launch Checklist
## From zero to live in one day

---

## BLOCK 1 — Get it live (2-3 hours)

### Step 1: Push to GitHub
```bash
cd ~/Desktop/ProjectX/files
git init
git add .
git commit -m "initial: GPU inference profiler v0.1"
gh repo create gpu-profiler --public --push
```
Done when: github.com/YOURNAME/gpu-profiler is publicly visible.

### Step 2: Deploy to Railway
1. Go to railway.app → New Project → Deploy from GitHub
2. Select your gpu-profiler repo
3. Set env var: ADMIN_SECRET = [random 20-char string]
4. Go to Settings → Networking → copy your public URL

Done when: YOUR_URL/health returns {"status": "ok"}

### Step 3: Run the tool yourself first
Open YOUR_URL in browser. Run this config:
- Model: meta-llama/Meta-Llama-3-8B-Instruct
- GPU: H100 SXM5
- Batch size: 1, Precision: FP16
- Flash Attention: OFF, Continuous batching: OFF
- GPU count: 4

Screenshot the result. This is your marketing asset.

### Step 4: Update your contact email
Search index.html and admin.py for "your@email.com" and replace with your real email.

---

## BLOCK 2 — First post (30 minutes)

### Step 5: Post on r/LocalLLaMA
See DISTRIBUTION.md for the exact post text.

### Step 6: Post in CUDA Mode Discord
discord.gg/cudamode — use the short version in DISTRIBUTION.md

### Step 7: Monitor first hour
```bash
cd ~/Desktop/ProjectX/files/backend
python admin.py stats
```

---

## BLOCK 3 — Follow up (daily from day 2)

### Step 8: Check leads every morning
```bash
python admin.py stats
python admin.py leads
python admin.py follow-up
```
Send personalized follow-up within 24 hours of every email captured.

### Step 9: Post Twitter thread (day 2)
See DISTRIBUTION.md section 4.

### Step 10: HuggingFace Discord + LinkedIn (day 3)
See DISTRIBUTION.md sections 3 and 5.

---

## BLOCK 4 — First customer (week 2-3)

### Step 11: Identify 3 warmest leads
Look for: left email + works at AI company + high savings estimate (>$10K/mo)

### Step 12: Convert one to paid pilot
Offer: 2-week manual optimization, $2,000 flat.
Guaranteed 20% efficiency improvement or full refund.

---

## Weekly targets

| Metric | Week 1 | Week 4 |
|---|---|---|
| Tool runs | 50 | 200 |
| Emails captured | 5 | 25 |
| Paid pilots | 0 | 1 |
| MRR | $0 | $2,000 |

If week 1 runs < 20: problem is distribution. Post in more communities.
If runs > 50 but emails < 5: problem is CTA. Make the offer more specific.
If emails > 10 but no pilots: follow up faster. Lower pilot price to $500.
