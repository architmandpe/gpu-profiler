# GPU Inference Profiler

Free LLM inference efficiency analysis tool. Enter a model + GPU, get a report showing
exactly where performance is being wasted and how much it's costing you.

## What it does

- Estimates throughput efficiency vs theoretical hardware peak
- Detects top bottlenecks: Flash Attention, batching, precision, KV cache, framework overhead
- Calculates estimated monthly GPU cost savings
- Works for all major open-source LLMs: Llama 3, Mistral, Mixtral, Qwen, Gemma, Phi, DeepSeek

---

## Setup (5 minutes)

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the server

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the tool

Navigate to http://localhost:8000 in your browser.

---

## Deploy to production

### Option A: Railway (fastest, free tier available)

```bash
# Install Railway CLI
npm install -g @railway/cli

# From project root
railway login
railway init
railway up
```

Railway auto-detects FastAPI and deploys. Set the start command to:
```
cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Option B: Render.com

1. Push to GitHub
2. Create a new Web Service on render.com
3. Build command: `pip install -r backend/requirements.txt`
4. Start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`

### Option C: VPS (DigitalOcean / Hetzner)

```bash
# On server
git clone your-repo
cd gpu-profiler/backend
pip install -r requirements.txt

# Run with process manager
pip install gunicorn
gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Set up nginx as reverse proxy (see nginx.conf)
```

### Option D: Docker

```bash
docker build -t gpu-profiler .
docker run -p 8000:8000 gpu-profiler
```

---

## Enabling real GPU profiling

The current version uses theoretical roofline calculations + benchmark baselines.
To enable actual GPU profiling (for when you have GPU hardware):

### Install GPU dependencies

```bash
pip install torch>=2.1.0 transformers>=4.40.0 flash-attn>=2.5.0 accelerate
```

### How real profiling works

The profiler module is designed to accept real measurements. When you have a GPU,
replace the `estimate_current_efficiency()` call in `profiler.py` with actual
PyTorch Profiler measurements:

```python
import torch
from torch.profiler import profile, ProfilerActivity

def run_real_benchmark(model, tokenizer, device, batch_size, context_length, num_runs=10):
    """Run actual forward passes and measure throughput."""
    inputs = tokenizer(
        ["Sample text"] * batch_size,
        return_tensors="pt",
        max_length=context_length,
        padding=True,
        truncation=True,
    ).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model.generate(**inputs, max_new_tokens=128)

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with torch.no_grad():
        for _ in range(num_runs):
            out = model.generate(**inputs, max_new_tokens=128)
    end.record()

    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    tokens_generated = num_runs * batch_size * 128
    tokens_per_sec = tokens_generated / (elapsed_ms / 1000)
    return tokens_per_sec
```

---

## Architecture

```
gpu-profiler/
├── backend/
│   ├── main.py           # FastAPI app + routes
│   ├── profiler.py       # Core analysis engine
│   ├── model_registry.py # LLM architecture specs
│   ├── gpu_specs.py      # GPU theoretical specs
│   └── requirements.txt
└── frontend/
    └── index.html        # Single-file frontend
```

The frontend is a single HTML file served by FastAPI. No build step, no npm, no webpack.
This is intentional — the goal is zero-friction deployment.

---

## Adding new models

Edit `backend/model_registry.py`. Add an entry to `MODEL_REGISTRY`:

```python
"org/model-name": {
    "display_name": "Model Display Name",
    "family": "family_name",
    "params_b": 7.0,          # parameter count in billions
    "num_layers": 32,
    "hidden_dim": 4096,
    "num_heads": 32,
    "num_kv_heads": 8,        # GQA heads (same as num_heads if MHA)
    "head_dim": 128,
    "vocab_size": 32000,
    "context_length": 4096,
    "architecture": "decoder-only",
    "attention_type": "GQA",
    "flash_attn_compatible": True,
    "fp16_weights_gb": 14.0,  # roughly: params_b * 2
    "min_vram_gb": 16,
    "recommended_vram_gb": 16,
},
```

Also add a shorthand alias to `MODEL_ALIASES` if you want user-friendly names.

---

## Adding new GPUs

Edit `backend/gpu_specs.py`. Add an entry to `GPU_SPECS`:

```python
"GPU Name": {
    "display_name": "Full GPU Name",
    "fp16_tflops": 989.4,          # From NVIDIA datasheet
    "memory_bandwidth_tbs": 3.35,  # In TB/s
    "vram_gb": 80,
    "generation": "hopper",
    "notes": "Description",
},
```

Also add a cost estimate to `GPU_HOURLY_RATES` in `profiler.py`.

---

## Go-to-market: how to use this tool

This tool is your top-of-funnel. The goal is not revenue from the tool itself.
The goal is warm leads.

### Week 1 distribution

Post in these communities with a screenshot of a real output:

1. **CUDA Mode Discord** — #general or #inference
2. **r/LocalLLaMA** — "We built a free tool to show you exactly how efficient your LLM stack is"
3. **Hugging Face Discord** — #speed-and-performance
4. **MLOps Community Slack**
5. **Twitter/X** — tag @marksaroufim @karpathy @pytorch

Post format:
> "We built a free tool that profiles your LLM inference stack and shows exactly
> where you're leaving performance on the table. Ran it on Llama 3 8B on H100 —
> found 3 critical bottlenecks, estimated 41% efficiency gap. Takes 10 seconds.
> [link]"

### Follow-up sequence

When someone uses the tool, reach out:
> "Hi [name], you analyzed [model] on [GPU] last week. We found [X]% efficiency gap.
> We've fixed this exact setup for 2 other companies. Would you spend 20 minutes
> to see how we'd approach it for you?"

### What you're selling after this

Not the tool. The optimization engagement:
- $2,000–$5,000 for a 30-day manual optimization
- $5,000–$12,000/month for ongoing automated optimization
- ROI framing: "We charge a fraction of what we save you"

---

## License

MIT — use this freely, build on it, improve it.
