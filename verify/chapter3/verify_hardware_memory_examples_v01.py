#!/usr/bin/env python3
"""Independently audit the textbook's KV-cache and MLA resource examples.

Only standard-library Python is required.  Values are logical tensor sizes,
not measurements of allocated GPU memory or end-to-end inference latency.

Textbook author / scientific question: Cheng-Rong Xie (原点者).
Reproducibility implementation: Wulan, 2026-11-12.
"""
from fractions import Fraction

GIB = 1024 ** 3


def dense_kv_bytes(batch, layers, tokens, kv_heads, head_dim, bytes_per_scalar):
    """Key and Value tensors, with no metadata or fragmentation overhead."""
    args = (batch, layers, tokens, kv_heads, head_dim, bytes_per_scalar)
    assert all(isinstance(v, int) and v > 0 for v in args)
    return 2 * batch * layers * tokens * kv_heads * head_dim * bytes_per_scalar


def latent_kv_bytes(batch, layers, tokens, content_dim, rope_dim, bytes_per_scalar):
    """MLA teaching model: content latent + separate positional key per position."""
    args = (batch, layers, tokens, content_dim, rope_dim, bytes_per_scalar)
    assert all(isinstance(v, int) and v > 0 for v in args)
    return batch * layers * tokens * (content_dim + rope_dim) * bytes_per_scalar


def main():
    # Chapter 3: 8 concurrent requests, 32 layers, 4096 tokens, 2-byte FP16.
    gqa = dense_kv_bytes(8, 32, 4096, 8, 128, 2)
    mha = dense_kv_bytes(8, 32, 4096, 32, 128, 2)
    assert gqa == 4 * GIB, (gqa, "expected 4 GiB")
    assert mha == 16 * GIB, (mha, "expected 16 GiB")

    # Mathematics appendix: batch=1, length=131072, 32 layers.
    mla = latent_kv_bytes(1, 32, 131072, 512, 64, 2)
    mha_reference = dense_kv_bytes(1, 32, 131072, 32, 128, 2)
    assert mla == Fraction(9, 2) * GIB, (mla, "expected 4.5 GiB")
    assert mha_reference == 64 * GIB, (mha_reference, "expected 64 GiB")

    ratio = Fraction(mha_reference, mla)
    assert ratio == Fraction(128, 9)
    assert f"{float(ratio):.3f}" == "14.222"

    # Simplified linear-layer weight-only arithmetic intensity: I=2B/s.
    assert Fraction(2 * 1, 2) == 1
    assert Fraction(2 * 32, 2) == 32

    print("PASS: chapter3 GQA KV = 4 GiB")
    print("PASS: chapter3 MHA KV = 16 GiB")
    print("PASS: appendix MLA latent KV = 4.500 GiB")
    print("PASS: appendix MHA reference KV = 64 GiB")
    print(f"PASS: MHA / MLA logical KV ratio = {float(ratio):.3f}")
    print("PASS: decode weight-only arithmetic intensity = 1, 32 FLOP/byte")
    print("Note: excludes auxiliary buffers, paging, fragmentation, quantization,")
    print("device communication, kernel launch costs, and latency measurements.")


if __name__ == "__main__":
    main()
