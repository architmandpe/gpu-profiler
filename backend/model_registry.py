"""
Registry of common open-source LLM architectures.
Used to calculate theoretical throughput ceilings and detect optimization opportunities.

Data sourced from: official model cards, Meta/Mistral/Google technical reports,
and community benchmarks on HuggingFace.
"""

MODEL_REGISTRY = {
    # ── Llama 3 family ──────────────────────────────────────────────────────
    "meta-llama/Meta-Llama-3-8B-Instruct": {
        "display_name": "Llama 3 8B Instruct",
        "family": "llama3",
        "params_b": 8.0,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,           # GQA
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 8192,
        "architecture": "decoder-only",
        "attention_type": "GQA",     # Grouped Query Attention
        "flash_attn_compatible": True,
        "fp16_weights_gb": 16.0,     # rough: 2 bytes * 8B params
        "min_vram_gb": 18,
        "recommended_vram_gb": 24,
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "display_name": "Llama 3 70B Instruct",
        "family": "llama3",
        "params_b": 70.0,
        "num_layers": 80,
        "hidden_dim": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 8192,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 140.0,
        "min_vram_gb": 160,
        "recommended_vram_gb": 160,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "display_name": "Llama 3.1 8B Instruct",
        "family": "llama3",
        "params_b": 8.0,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 131072,    # 128k context
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 16.0,
        "min_vram_gb": 18,
        "recommended_vram_gb": 24,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "display_name": "Llama 3.1 70B Instruct",
        "family": "llama3",
        "params_b": 70.0,
        "num_layers": 80,
        "hidden_dim": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 140.0,
        "min_vram_gb": 160,
        "recommended_vram_gb": 160,
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "display_name": "Llama 3.2 3B Instruct",
        "family": "llama3",
        "params_b": 3.0,
        "num_layers": 28,
        "hidden_dim": 3072,
        "num_heads": 24,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 6.0,
        "min_vram_gb": 8,
        "recommended_vram_gb": 8,
    },
    # ── Mistral / Mixtral ────────────────────────────────────────────────────
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "display_name": "Mistral 7B Instruct v0.3",
        "family": "mistral",
        "params_b": 7.3,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 32768,
        "context_length": 32768,
        "architecture": "decoder-only",
        "attention_type": "GQA+SWA",  # Sliding Window Attention
        "flash_attn_compatible": True,
        "fp16_weights_gb": 14.6,
        "min_vram_gb": 16,
        "recommended_vram_gb": 16,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "display_name": "Mixtral 8x7B Instruct",
        "family": "mixtral",
        "params_b": 46.7,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 32768,
        "context_length": 32768,
        "architecture": "MoE",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 93.0,
        "min_vram_gb": 96,
        "recommended_vram_gb": 96,
        "active_params_b": 12.9,     # only 2 of 8 experts active per token
    },
    # ── Qwen 2.5 ────────────────────────────────────────────────────────────
    "Qwen/Qwen2.5-7B-Instruct": {
        "display_name": "Qwen 2.5 7B Instruct",
        "family": "qwen2",
        "params_b": 7.6,
        "num_layers": 28,
        "hidden_dim": 3584,
        "num_heads": 28,
        "num_kv_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 15.2,
        "min_vram_gb": 16,
        "recommended_vram_gb": 24,
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "display_name": "Qwen 2.5 72B Instruct",
        "family": "qwen2",
        "params_b": 72.7,
        "num_layers": 80,
        "hidden_dim": 8192,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 152064,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 145.4,
        "min_vram_gb": 160,
        "recommended_vram_gb": 160,
    },
    # ── Gemma 2 ─────────────────────────────────────────────────────────────
    "google/gemma-2-9b-it": {
        "display_name": "Gemma 2 9B Instruct",
        "family": "gemma2",
        "params_b": 9.2,
        "num_layers": 42,
        "hidden_dim": 3584,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 256,
        "vocab_size": 256000,
        "context_length": 8192,
        "architecture": "decoder-only",
        "attention_type": "GQA+SA",   # Sliding + Local attention
        "flash_attn_compatible": True,
        "fp16_weights_gb": 18.4,
        "min_vram_gb": 20,
        "recommended_vram_gb": 24,
    },
    "google/gemma-2-27b-it": {
        "display_name": "Gemma 2 27B Instruct",
        "family": "gemma2",
        "params_b": 27.2,
        "num_layers": 46,
        "hidden_dim": 4608,
        "num_heads": 32,
        "num_kv_heads": 16,
        "head_dim": 128,
        "vocab_size": 256000,
        "context_length": 8192,
        "architecture": "decoder-only",
        "attention_type": "GQA+SA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 54.4,
        "min_vram_gb": 60,
        "recommended_vram_gb": 80,
    },
    # ── Phi-3 / Phi-3.5 ─────────────────────────────────────────────────────
    "microsoft/Phi-3.5-mini-instruct": {
        "display_name": "Phi-3.5 Mini Instruct",
        "family": "phi3",
        "params_b": 3.8,
        "num_layers": 32,
        "hidden_dim": 3072,
        "num_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 96,
        "vocab_size": 32064,
        "context_length": 128000,
        "architecture": "decoder-only",
        "attention_type": "MHA",      # Multi-Head Attention (no GQA)
        "flash_attn_compatible": True,
        "fp16_weights_gb": 7.6,
        "min_vram_gb": 8,
        "recommended_vram_gb": 8,
    },
    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "display_name": "DeepSeek R1 Distill Llama 8B",
        "family": "deepseek",
        "params_b": 8.0,
        "num_layers": 32,
        "hidden_dim": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 16.0,
        "min_vram_gb": 18,
        "recommended_vram_gb": 24,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": {
        "display_name": "DeepSeek R1 Distill Qwen 32B",
        "family": "deepseek",
        "params_b": 32.8,
        "num_layers": 64,
        "hidden_dim": 5120,
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 152064,
        "context_length": 131072,
        "architecture": "decoder-only",
        "attention_type": "GQA",
        "flash_attn_compatible": True,
        "fp16_weights_gb": 65.6,
        "min_vram_gb": 80,
        "recommended_vram_gb": 80,
    },
}

# Common shorthand aliases -> canonical HF model IDs
MODEL_ALIASES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama 3 8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3 8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama 3.1 8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama 3 70b": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama 3.2 3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral 7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mixtral 8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen 2.5 7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "gemma-2-9b": "google/gemma-2-9b-it",
    "gemma 2 9b": "google/gemma-2-9b-it",
    "gemma-2-27b": "google/gemma-2-27b-it",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "deepseek-r1-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek r1 8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

def resolve_model(name: str) -> tuple[str | None, dict | None]:
    """
    Resolve a model name (alias or full HF ID) to its registry entry.
    Returns (canonical_id, spec_dict) or (None, None) if not found.
    """
    name_clean = name.strip().lower()

    # Try direct lookup
    for key, spec in MODEL_REGISTRY.items():
        if key.lower() == name_clean:
            return key, spec

    # Try alias lookup
    canonical = MODEL_ALIASES.get(name_clean)
    if canonical:
        return canonical, MODEL_REGISTRY.get(canonical)

    # Fuzzy: check if the cleaned name is a substring of any key
    for key, spec in MODEL_REGISTRY.items():
        if name_clean in key.lower() or key.lower() in name_clean:
            return key, spec

    return None, None

def get_model_list() -> list[str]:
    """Return display names for the UI dropdown."""
    return [v["display_name"] for v in MODEL_REGISTRY.values()]

def get_model_ids() -> list[str]:
    return list(MODEL_REGISTRY.keys())
