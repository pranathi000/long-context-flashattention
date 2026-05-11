# Memory-Efficient Long-Context Transformer Attention using Paged KV-Cache and IO-Aware FlashAttention-Style Execution

## Project Overview

This project implements a memory-efficient long-context transformer inference pipeline inspired by FlashAttention and modern LLM inference systems.

The primary goal of the project is to solve the core bottlenecks that appear in large-scale transformer inference:

- Quadratic attention memory scaling
- Long-context inference inefficiency
- KV-cache fragmentation and allocation overhead
- Numerical instability during streaming softmax computation
- Memory explosion during large-context execution

The implementation progressively evolved from:

1. Naive full attention
2. FlashAttention-style tiled attention
3. Approximate chunked attention
4. Exact streaming FlashAttention
5. Paged KV-cache management
6. Correctness validation against baseline full attention
7. Long-context memory scaling experiments

---

# Core Problem

## Why Naive Attention Fails

Standard transformer attention computes:

\[
O = softmax\left(\frac{QK^T}{\sqrt{d}}\right)V
\]

The attention matrix has shape:

\[
N \times N
\]

where:

- N = sequence length

This causes:

- Quadratic memory growth
- Huge GPU memory overhead
- Infeasible long-context inference

Example:

| Context Length | Attention Matrix Size |
|---|---|
| 4K | ~16M entries |
| 32K | ~1B entries |
| 128K | ~17B entries |

---

# Main Objective

The project reproduces core ideas used in:

- FlashAttention
- PagedAttention
- Streaming transformer execution
- Long-context LLM inference systems

The focus is on:

- IO-aware execution
- Streaming softmax computation
- Tiled attention computation
- Memory-efficient inference
- Numerical correctness preservation

---

# Project Structure

```text
LONG CONTEXT ATTENTION/
│
├── flash_attention/
│   ├── naive_attention.py
│   ├── tiled_attention.py
│   └── flashattention_utils.py
│
├── paged_kv_cache/
│   ├── allocator.py
│   ├── page_table.py
│   └── reuse_policy.py
│
├── chunked_attention/
│   ├── chunk_executor.py
│   └── streaming_attention.py
│
├── experiments/
│   ├── memory_scaling.py
│   ├── latency_scaling.py
│   ├── correctness_validation.py
│   └── plot_results.py
│
├── assets/
│   ├── memory_scaling_plot.png
│   ├── latency_scaling_plot.png
│   └── memory_reduction_plot.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# File-by-File Breakdown

# 1. naive_attention.py

## Purpose

Implements baseline full transformer attention.

## Features

- Full QKᵀ computation
- Global softmax normalization
- Exact attention reference implementation

## Complexity

| Metric | Complexity |
|---|---|
| Memory | O(N²) |
| Compute | O(N²) |

---

# 2. tiled_attention.py

## Purpose

Implements FlashAttention-style tiled attention execution.

## Key Idea

Instead of materializing the entire attention matrix:

- break K/V into tiles
- process one tile at a time
- stream computation through GPU memory

## Features

- Tile-wise QK computation
- Running-max stabilization
- Running-sum normalization
- Streaming softmax computation

## Benefit

Reduces active memory footprint dramatically.

---

# 3. flashattention_utils.py

## Purpose

Contains helper functions for streaming FlashAttention.

## Features

### Running Max Update

Maintains numerical stability during streaming softmax accumulation.

### Running Sum Update

Maintains correct online softmax denominator.

### Streaming Exponential Rescaling

Ensures exact normalization when new maxima appear.

---

# 4. allocator.py

## Purpose

Implements paged KV-cache allocation.

## Features

- Block-based page allocation
- Dynamic page assignment
- Free page tracking
- Memory usage accounting

## Why It Matters

Prevents:

- memory fragmentation
- inefficient allocations
- large contiguous memory requirements

---

# 5. page_table.py

## Purpose

Implements logical-to-physical KV page mapping.

## Features

- Logical page abstraction
- Physical page lookup
- Token-to-page translation
- Offset computation

---

# 6. reuse_policy.py

## Purpose

Implements KV-cache reuse and memory reclamation policy.

## Features

- Page reuse
- Free-page pool
- Request completion handling
- Dynamic memory recycling

---

# 7. chunk_executor.py

## Purpose

Initial attempt at chunked long-context execution.

## Important Discovery

This implementation FAILED correctness validation.

## Why It Failed

Softmax normalization was computed independently per chunk.

This broke:

- global attention normalization
- exact attention semantics

## Engineering Lesson

Softmax is NOT separable across chunks.

This directly motivated:
- exact streaming attention

---

# 8. streaming_attention.py

## Purpose

Implements mathematically exact streaming FlashAttention.

## Main Achievement

This implementation preserves:

- exact global softmax normalization
- bounded memory execution
- streaming accumulation correctness

## Core Algorithm

Maintains globally:

- running max
- running sum
- running output accumulation

while streaming through KV tiles.

## Key Formula

When new maxima appear:

\[
m_{new} = max(m_{old}, m_{tile})
\]

Old accumulators are rescaled:

\[
l_{new} = e^{m_{old}-m_{new}}l_{old} + \sum e^{s-m_{new}}
\]

This enables:

- exact online softmax
- numerical stability
- streaming execution

---

# 9. memory_scaling.py

## Purpose

Benchmarks GPU memory usage across sequence lengths.

## Compared Methods

- Naive Attention
- Tiled FlashAttention
- Streaming Attention

## Metrics

- Peak GPU memory
- Sequence length scaling
- Memory reduction factor

---

# 10. latency_scaling.py

## Purpose

Measures runtime latency across increasing sequence lengths.

## Insight

Streaming attention trades:

- additional compute overhead

for:

- massive memory savings

This is the core FlashAttention tradeoff.

---

# 11. correctness_validation.py

## Purpose

Validates numerical correctness against naive full attention.

## Validation Metrics

- Mean Absolute Error
- Max Absolute Error
- Mean Relative Error

## Final Result

```text
Mean Absolute Error ≈ 1e-8
```

This demonstrated:

- numerical equivalence
- mathematically exact streaming attention
- correct online softmax implementation

---

# 12. plot_results.py

## Purpose

Generates benchmark visualizations.

## Generated Plots

### Memory Scaling Plot

Shows:
- naive attention memory explosion
- bounded memory growth of streaming attention

### Latency Scaling Plot

Shows:
- runtime overhead tradeoffs
- scaling behavior

### Memory Reduction Plot

Shows:
- increasing efficiency advantage over naive attention

---

# Main Technical Achievements

## 1. Exact Streaming FlashAttention

Implemented mathematically exact streaming attention with:

- running-max stabilization
- running-sum normalization
- online softmax accumulation

---

## 2. Bounded Memory Execution

Avoided materializing:

- full attention matrices
- massive intermediate tensors

This enabled scalable long-context inference.

---

## 3. Numerical Correctness Validation

Validated implementation against naive attention with:

```text
~1e-8 mean absolute error
```

This proved:

- correctness
- stability
- exact equivalence

---

## 4. Paged KV-Cache Management

Implemented:

- page allocation
- logical-to-physical mapping
- page reuse policies

Inspired by modern inference runtimes.

---

# Experimental Results

## Memory Scaling

Observed:

- Naive attention memory grows rapidly
- Streaming attention memory remains bounded
- Significant memory reduction achieved

## Correctness

| Method | Validation |
|---|---|
| Tiled FlashAttention | PASSED |
| Streaming FlashAttention | PASSED |
| Chunked Approximation | FAILED |

This demonstrates:
- exact streaming implementation correctness
- importance of global online softmax

---

# Key Engineering Lessons

## 1. Chunked Softmax Is Incorrect

Independent chunk normalization breaks:
- global attention semantics
- exact softmax behavior

---

## 2. Streaming Online Softmax Solves The Problem

Maintaining:
- running max
- running sum
- running output

preserves exact attention equivalence.

---

## 3. Memory Efficiency Requires IO Awareness

Reducing:
- DRAM traffic
- attention materialization
- temporary tensor creation

is critical for scalable inference.

---

# Technologies Used

## Frameworks

- PyTorch
- CUDA
- Matplotlib
- Pandas

## Concepts

- FlashAttention
- Transformer Attention
- Streaming Softmax
- Online Normalization
- KV-Cache Paging
- Long-Context Inference
- GPU Memory Optimization
- IO-Aware Execution

---

# Future Improvements

Potential future extensions:

- Triton kernel implementation
- CUDA fused kernels
- Block-sparse attention
- Multi-head attention support
- FP16/BF16 mixed precision
- Quantized KV-cache
- Speculative decoding integration
- Continuous batching
- Production inference server integration

---

# Final Resume Summary

Implemented mathematically exact FlashAttention-style streaming attention with paged KV-cache management, achieving significant memory reduction compared to naive attention while preserving numerical equivalence during scalable long-context transformer inference.

---

# Author

Santhoshini
