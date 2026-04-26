# Distribution Posts — Copy-Paste Ready

Replace [https://gpu-inference-profiler-production.up.railway.app] with your deployed Railway URL before posting.
Take a screenshot of Llama 3 8B on H100 with default settings first — that's your image.

---

## 1. r/LocalLLaMA

**Title:** We built a free tool that shows exactly how efficient your LLM inference stack is

We have been doing inference optimization work and kept seeing the same thing: most
teams running LLMs in production leave 30-60% of GPU performance on the floor but
don't know exactly where.

So we built a free profiler. Enter your model + GPU + config and get:
- Throughput vs theoretical hardware peak
- Memory bandwidth utilization
- Top 3 bottlenecks ranked by impact
- Estimated monthly savings if you fix them

We ran it on Llama 3 8B on H100 SXM5 with stock config (batch=1, no Flash Attention):
- 34% efficiency score
- 90 tok/s vs 234 tok/s theoretical peak
- Bottleneck 1: No Flash Attention 2 — 2x speedup available
- Bottleneck 2: Batch size 1 — 3-8x throughput improvement possible
- Estimated $5,900/month in avoidable GPU spend (4x H100)

Free, no signup: [https://gpu-inference-profiler-production.up.railway.app]

---

## 2. CUDA Mode Discord

Shipped something useful — free LLM inference profiler.

Give it a model + GPU, get a roofline analysis showing your efficiency gap + top bottlenecks.

Llama 3 8B / H100 SXM5 unoptimized: 90 tok/s measured vs 234 tok/s peak.
Detects missing FA2, batch_size=1, precision opportunities, KV cache issues.

[https://gpu-inference-profiler-production.up.railway.app] — free, no signup. Feedback welcome especially from people who have
profiled these models for real.

---

## 3. HuggingFace Discord (#speed-and-performance)

Free inference efficiency analyzer — input model + GPU + config, output efficiency
score vs theoretical peak, top bottlenecks, estimated cost gap.

Supports Llama 3/3.1/3.2, Mistral, Mixtral, Qwen 2.5, Gemma 2, Phi-3.5, DeepSeek R1
across H100/A100/L40S/A10G/RTX 4090.

[https://gpu-inference-profiler-production.up.railway.app] — free, no login

---

## 4. Twitter/X thread

Tweet 1:
We profiled Llama 3 8B on H100 with a typical unoptimized setup.
Result: 34% efficiency. 90 tok/s when the hardware can do 234 tok/s.
Here is exactly what is costing you performance [screenshot]

Tweet 2:
Bottleneck 1: No Flash Attention 2
FA2 is free to install and gives ~2x throughput.
Most teams in production are not using it.

Tweet 3:
Bottleneck 2: batch_size=1
GPU processes one request at a time. Continuous batching fills it dynamically.
3-8x throughput improvement. Zero accuracy impact.

Tweet 4:
Bottleneck 3: FP16 when INT8 works
Llama 8B in FP16 = 16GB weights. INT8 = 8GB. Same quality. 1.5x more throughput.

Tweet 5:
We built a free tool to run this on your own model and GPU in 10 seconds.
No signup: [https://gpu-inference-profiler-production.up.railway.app]
cc @marksaroufim @pytorch

---

## 5. LinkedIn

Most AI companies run GPUs at 35-60% of theoretical peak.
Not bad hardware. Bad software tuning.

We built a free profiler: input model + GPU + config, get efficiency score,
top bottlenecks, and estimated monthly savings.

Llama 3 8B / H100 SXM5 untuned: 90 tok/s vs 234 tok/s peak.
Top issue: Flash Attention 2 not enabled. Free to fix. 2x speedup available.

Free at [https://gpu-inference-profiler-production.up.railway.app] — no login required.

---

## Timing

Day 1: r/LocalLLaMA + CUDA Mode Discord
Day 2: Twitter thread
Day 3: HuggingFace Discord + LinkedIn
Day 5: Follow up in CUDA Mode with before/after benchmark numbers

Goal week 1: 50 tool runs
Goal week 2: 3 warm leads who reply to follow-up email
