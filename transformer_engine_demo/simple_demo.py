""" 
To create a **Minimal Working Example (MWE)** that demonstrates a **NVIDIA NeMo-enhanced Transformer** using **Transformer Engine** with **sliding window attention**, here’s a step-by-step guide and a code sample. This will use NeMo’s integration with Transformer Engine and show how to configure sliding window attention.

---

## 1. Prerequisites

- **NVIDIA NeMo** (latest, with Transformer Engine support)
- **Transformer Engine** (`pip install transformer-engine`)
- **PyTorch** (compatible version)
- **CUDA** (for GPU, optional for CPU-only demo)

---

## 2. Key Concepts

- **Transformer Engine**: NVIDIA’s library for fast, memory-efficient transformer layers.
- **Sliding Window Attention**: Restricts attention to a local window, reducing memory and compute.

---

## 3. Example: NeMo Transformer with Sliding Window Attention

Below is a minimal script using NeMo’s `nemo.collections.nlp.modules.common.megatron.transformer` with Transformer Engine and sliding window attention.

---

## 4. Explanation

- **attn_mask_type="local"**: This enables sliding window attention in NeMo.
- **window_size**: Controls the size of the local window.
- **transformer_engine=True**: Uses NVIDIA’s Transformer Engine for the layer.
- **fp8**: Optional, for FP8 precision (requires Hopper GPUs).

---

## 5. Notes

- This MWE uses a single `TransformerLayer` for simplicity. For a full model, stack multiple layers.
- You can adjust `hidden_size`, `num_attention_heads`, and `window_size` as needed.
- Make sure your CUDA and PyTorch versions are compatible with Transformer Engine.

---

## 6. References

- [NeMo TransformerLayer Docs](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/modules.html#transformerlayer)
- [Transformer Engine GitHub](https://github.com/NVIDIA/TransformerEngine)
- [Sliding Window Attention in NeMo](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/models.html#sliding-window-attention)

---

**Let me know if you want a full training loop, dataset integration, or a multi-layer transformer example!**
"""
import torch
from nemo.collections.nlp.modules.common.megatron.transformer import TransformerLayer
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

# Ensure Transformer Engine is available
apex_guard = ApexGuardDefaults()
if not apex_guard.has_transformer_engine:
    raise ImportError("Transformer Engine is not installed or not found.")


def create_sliding_window_transformer(
    hidden_size: int = 256,
    num_attention_heads: int = 4,
    window_size: int = 8,
    seq_length: int = 32,
    use_fp8: bool = False,
):
    """Create a NeMo TransformerLayer with sliding window attention using Transformer Engine.

    Args:
        hidden_size: Transformer hidden size.
        num_attention_heads: Number of attention heads.
        window_size: Size of the sliding window.
        seq_length: Input sequence length.
        use_fp8: Whether to use FP8 precision (requires supported hardware).

    Returns:
        Output tensor after passing through the transformer layer.

    """
    # Dummy input
    x = torch.randn(seq_length, 1, hidden_size).cuda()

    # Create TransformerLayer with sliding window attention
    layer = TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=num_attention_heads,
        attention_type="scaled_dot_product",
        attn_mask_type="local",  # Enables sliding window
        window_size=window_size,
        transformer_engine=True,
        fp8=use_fp8,
    ).cuda()

    # Forward pass
    output = layer(x)
    return output


if __name__ == "__main__":
    out = create_sliding_window_transformer()
    print(f"Output shape: {out.shape}")
    print("Sliding window attention with Transformer Engine works!")
