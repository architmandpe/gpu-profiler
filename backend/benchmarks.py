"""
Benchmark baselines stub.
Real benchmarks added when GPU hardware is available.
Until then, profiler falls back to roofline estimates.
"""

def get_real_benchmark(model_family=None, params_b=None, gpu_name=None,
                       batch_size=1, precision="fp16", framework="vllm", **kwargs):
    """
    Returns real measured benchmark dict if available, else None.
    Format: {"tps": float, "source": str}
    """
    return None


def get_optimized_target(model_family, params_b, gpu_name, precision="fp16"):
    """
    Returns best achievable tokens/sec for this config based on known benchmarks.
    Returns None to let profiler use its own roofline estimate.
    """
    return None
