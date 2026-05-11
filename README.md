# Long-Context FlashAttention Inference

Memory-efficient long-context transformer inference using:

- FlashAttention-style tiled execution
- Online streaming softmax
- Paged KV-cache management
- Exact streaming attention
- Long-context scaling benchmarks
- Numerical correctness validation

## Features

- Exact streaming softmax accumulation
- Running-max stabilization
- Running-sum normalization
- KV-cache paging and reuse
- Memory scaling analysis
- Correctness validation against naive attention

## Benchmarks

Demonstrated scalable long-context inference with significantly reduced memory growth compared to naive attention.

## Author

Pranav
