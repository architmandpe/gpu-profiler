"""
GPU theoretical peak specifications.
Sources: NVIDIA official datasheets, MLPerf benchmarks.
"""

GPU_SPECS = {
    "H100 SXM5": {
        "display_name": "NVIDIA H100 SXM5",
        "fp16_tflops": 989.4,
        "memory_bandwidth_tbs": 3.35,
        "vram_gb": 80,
        "generation": "hopper",
        "notes": "Best-in-class for LLM inference at scale",
    },
    "H100 PCIe": {
        "display_name": "NVIDIA H100 PCIe",
        "fp16_tflops": 756.0,
        "memory_bandwidth_tbs": 2.0,
        "vram_gb": 80,
        "generation": "hopper",
        "notes": "PCIe variant, ~40% lower bandwidth than SXM5",
    },
    "A100 SXM4 80GB": {
        "display_name": "NVIDIA A100 SXM4 80GB",
        "fp16_tflops": 312.0,
        "memory_bandwidth_tbs": 2.0,
        "vram_gb": 80,
        "generation": "ampere",
        "notes": "Workhorse of enterprise AI inference",
    },
    "A100 PCIe 80GB": {
        "display_name": "NVIDIA A100 PCIe 80GB",
        "fp16_tflops": 312.0,
        "memory_bandwidth_tbs": 1.935,
        "vram_gb": 80,
        "generation": "ampere",
        "notes": "PCIe variant of A100",
    },
    "A100 40GB": {
        "display_name": "NVIDIA A100 40GB",
        "fp16_tflops": 312.0,
        "memory_bandwidth_tbs": 1.555,
        "vram_gb": 40,
        "generation": "ampere",
        "notes": "Smaller VRAM variant",
    },
    "H200 SXM": {
        "display_name": "NVIDIA H200 SXM",
        "fp16_tflops": 989.4,
        "memory_bandwidth_tbs": 4.8,
        "vram_gb": 141,
        "generation": "hopper",
        "notes": "Higher bandwidth successor to H100 SXM5",
    },
    "L40S": {
        "display_name": "NVIDIA L40S",
        "fp16_tflops": 733.0,
        "memory_bandwidth_tbs": 0.864,
        "vram_gb": 48,
        "generation": "ada",
        "notes": "Inference-optimized, cost-effective for smaller models",
    },
    "A10G": {
        "display_name": "NVIDIA A10G",
        "fp16_tflops": 125.0,
        "memory_bandwidth_tbs": 0.6,
        "vram_gb": 24,
        "generation": "ampere",
        "notes": "Common in AWS g5 instances",
    },
    "RTX 4090": {
        "display_name": "NVIDIA RTX 4090",
        "fp16_tflops": 330.0,
        "memory_bandwidth_tbs": 1.008,
        "vram_gb": 24,
        "generation": "ada",
        "notes": "Consumer GPU, popular for local inference",
    },
    "RTX 3090": {
        "display_name": "NVIDIA RTX 3090",
        "fp16_tflops": 142.0,
        "memory_bandwidth_tbs": 0.936,
        "vram_gb": 24,
        "generation": "ampere",
        "notes": "Previous gen consumer, still widely used",
    },
}

def get_gpu_list():
    return list(GPU_SPECS.keys())

def get_gpu_spec(gpu_name: str):
    return GPU_SPECS.get(gpu_name)
