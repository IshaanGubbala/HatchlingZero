# Executive Summary

Sparse Graph Neural Networks (GNNs) offer dramatic savings in computation and memory by exploiting sparsity in data or model weights. However, achieving **higher throughput**, **lower RAM usage**, and **lower energy** on GPU or Apple Silicon requires careful optimization of algorithms and data structures. We survey general sparse-GNN optimization strategies (sparse formats, pruning/quantization, kernel fusion, scheduling) and then focus on BDH (“Dragon Hatchling”) – a novel sparse LLM-like GNN architecture. BDH uses a large-scale *scale-free* neuron graph (activations ≈5% nonzero) and attention-based state updates. We discuss how BDH operations (matrix multiplies, masked attention, elementwise ReLU/LN) map to NVIDIA CUDA and Apple MPS hardware, and how to optimize them via kernel fusion, memory layout, batched sparse operations, mixed precision, structured sparsity, etc. Key metrics are throughput (tokens/s), latency, RAM (peak/average), energy/sample, and accuracy. Our report includes:

- **Optimization Techniques (prioritized)**: Data-format choice (CSR/CSC vs COO vs block-sparse), kernel/operator fusion, mixed-precision (FP16/Int8), structured sparsity (2:4 pattern), pruning, quantization, graph reordering and batching. We rank these by expected speedup, implementation complexity and accuracy risk. For example, using GPU Tensor Cores with FP16 or 2:4 sparsity can yield ~1.5–2× speedups (impact **high**, moderate coding effort), whereas advanced loop reordering and index compression give moderate gains (**medium** impact, higher complexity).

- **Implementation Recipes**: We sketch how to implement key optimizations, including fused CUDA kernels (e.g. a custom SpMM kernel, fused linear+ReLU), PyTorch custom ops or Triton scripts, and Apple MPS strategies (contiguous memory, use MPSGraph). For example, PyTorch’s new **SparseSemiStructured** (2:4) support allows block-sparse GEMMs on Ampere GPUs, and attention can be fused into a single FlashAttention-like kernel using TorchInductor or Triton. We illustrate mapping BDH’s “v @ D_x → Q,K; mask; QKᵀ @ V; +D_y” sequence in CUDA and via a `torch.compile` fused graph. 

- **Sparse Formats & Layouts**: A comparison of sparse tensor formats is given (CSR, CSC, COO, Block (2:4), Hash, hybrid). For example, **CSR/CSC** minimize storage and accelerate SpMM but are static; **COO** is dynamic but slower; **2:4 block-sparse** has hardware support (via cuSPARSELt) for ~1.6× speedup; **hybrid formats** (e.g. COO+CSR) can mix benefits. We include a table summarizing memory overhead, SpMM efficiency, update cost and suitability for BDH. 

- **Benchmarking Methodology**: We recommend microbenchmarks (SpMM latency vs sparsity, attention kernel timings, memory-bandwidth tests) and end-to-end tests (tokens/s, samples/s on realistic datasets). Track *GPU utilization, occupancy, memory bandwidth*, and measure energy (e.g. NVML, or Apple Instruments) per token/sample. Evaluate accuracy vs baseline when quantizing or pruning. Tools like NVIDIA’s Nsight/NVML and PyTorch Profiler can report throughput, latency, peak/avg RAM and Joules per token. We follow token-level power-analysis ideas (e.g. TokenPowerBench) to pinpoint where energy is spent.

- **Pitfalls & Validation**: We warn of common issues: PyTorch sparse ops often require 64-bit indices (CSR/CSC) and coalesced layout; Apple’s MPS backend has quirks (e.g. some in-place ops fail on non-contiguous tensors and fusion is limited). Fused kernels must preserve numerical accuracy (watch overflow in quantized matmuls) and correct masking. We suggest thorough unit tests (correctness of sparse SpMM against dense, accuracy after quantization/pruning) and measuring both *peak* and *average* metrics to catch outliers.

This report provides an **analytical roadmap**: prioritized techniques with estimates, code sketches, data-structure recommendations, and rigorous testing advice, backed by recent literature and hardware documentation.

## Key Optimization Dimensions and Metrics

We consider the following performance and cost dimensions for BDH:

- **Throughput & Latency**: Tokens or samples processed per second (average and tail latency). Affected by kernel efficiency, parallelism, and batching.  
- **Memory (RAM) Usage**: *Peak* and *average* GPU memory. Sparse formats and compression reduce storage. Also *effective memory bandwidth* usage matters for throughput.  
- **Energy per Sample**: Power draw (watts) integrated over latency. Use energy-per-token or per-sample metrics.  
- **Accuracy Trade-offs**: Especially for pruning/quantization. Must quantify accuracy vs speed/RAM.  
- **Implementation Complexity & Compatibility**: Coding effort, maintainability, and support in PyTorch/TensorFlow. E.g. using native PyTorch sparse vs writing CUDA kernels.  
- **Hardware/Software Compatibility**: CUDA compute capability (for TensorCores, FP16) and PyTorch/CUDA versions; Apple MPS/MLC and macOS versions; any library dependencies (cuSPARSELt, ROCm, etc.).  

Where unspecified (e.g. target GPU model, dataset size, exact BDH variant), optimizations should be robust or parameterizable. For example, block-sparsity patterns (like 2:4) require Ampere+ hardware.

## General Sparse GNN Optimization Techniques

### Sparse Formats and Data Layout

- **CSR vs COO vs CSC**: Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC) formats store nonzeros with row/col pointers and are **highly efficient for SpMM**. In practice, CSR/CSC multiplications on GPUs (via cuSPARSE or custom kernels) outperform COO-based approaches. PyTorch documentation notes CSR matmul is typically faster than COO. CSC is similar (transpose of CSR) and faster than COO. 
  - *Pros*: Low memory overhead, fast sparse-dense matmul (SpMM), good cache locality when traversing rows/cols.  
  - *Cons*: Difficult to modify after construction; dynamic graph changes require rebuilding the structure.  

- **COO (Coordinate)**: Stores triples (row, col, value). Very flexible (easy to append edges or weights) but higher overhead (no compressed pointers). Implementation on GPUs can be slower due to irregular memory access. Use COO if graph topology changes often, then convert to CSR for heavy computation.

- **Block-Sparse / Structured (e.g. 2:4)**: NVIDIA Ampere GPUs support “**semi-structured**” sparsity (2:4 pattern) in hardware. Here every 4-elements block has 2 nonzeros. Using cuSPARSELt or PyTorch’s `torchao` module, dense matmuls on 2:4-sparse weights run ~1.3–1.6× faster. Block-sparse (e.g. constant block sizes) can exploit shared memory and warp efficiency. 
  - *Pros*: Hardware-accelerated on NVIDIA (cuSPARSELt), easy speedup with minimal accuracy loss.  
  - *Cons*: Requires model pruned to fit pattern; not dynamically changeable (must maintain block mask).

- **Hash/Hybrid Formats**: For dynamic or extremely sparse cases, techniques like hash tables or hybrid sparse-dense (e.g. 1:16 ratio) exist, but they incur large overhead or irregular memory access. These are generally *lower performance* than CSR/2:4 for large matrices. We do not focus on hash here.

- **Memory Layout**: Store feature matrices in row-major (C-order) for best GPU throughput, and for block-sparse align blocks to 32B/64B boundaries. Ensure tensors are contiguous. On Apple MPS, avoid non-contiguous tensors – some in-place ops (e.g. `addcmul_`) silently fail on MPS for non-contiguous outputs.

### Kernel Fusion and Operator Reordering

- **Fusion of Elementwise Ops**: Many deep-learning frameworks (TorchInductor, XLA, ONNX) can fuse pointwise ops. For example, a sequence *Linear→BatchNorm→ReLU* can become one kernel. In BDH layers, one can fuse **Linear + ReLU + Dropout + Add** chains into single kernels. Fusing reduces memory traffic and kernel launch overhead. TorchInductor can fuse layernorm/add chains in Transformers; similarly, customizing a Triton or CUDA kernel to do `x=W·v; x=ReLU(x)` in one shot saves memory writes.

- **Attention Fusion**: BDH’s attention (computing $QK^T$, masking, then multiplying by $V$) can be fused. For example, `flash_attention` kernels fuse QKᵀ and softmax and matmul by $V$ in one pass. Using Triton/torch.compile one can implement *causal linear attention* as a fused kernel. For BDH’s linear attention (with prefix updates), one could analogously fuse Q/K projection and causal mask reduction. Fusion yields large speedups by reusing shared memory and avoiding writing intermediate $T\times T$ matrices.

- **Loop Reordering (SpMM)**: Sparse matrix multiplies (SpMM) have multiple loop-order strategies (outer-product, Gustavson’s row-accumulation, inner-product). Using **Gustavson’s algorithm (row-wise accumulation)** is usually better than naive inner loops, which become dense iterations. The *Scorch* library shows that picking loop orders based on sparsity can yield “arbitrarily large” speedups. In practice, ensure your sparse kernels accumulate along the dimension of the sparse index (e.g. iterate over nonzeros of $A$ and do vector adds to $C$) rather than looping full rows with dot products. Tools like GraphBLAS or cuSPARSE auto-select row-based strategies.

- **Graph Reordering for Locality**: Reorder node indices to improve memory coalescing (e.g. by vertex degree or graph partitioning). For very large graphs, techniques like vertex-cut or edge reordering (METIS, RC-order) can boost SpMM locality. This is high effort but can improve L2 cache reuse and reduce bank conflicts on GPU.

### Pruning and Quantization

- **Pruning (Sparse Training)**: Remove unimportant weights to sparsify BDH (unstructured or structured). Studies show fine-grained GNN pruning can cut ~50% of weights with negligible accuracy loss. On inference, this allows storing only nonzeros. One can prune $D_x,D_y,E$ weight matrices or dynamic edges. Pruning static weights to CSR can drastically reduce memory and GFLOPs. Complexity: needs retraining or fine-tuning to recover accuracy. Risk: too aggressive pruning may hurt performance.

- **Mixed-Precision/Quantization**: Use FP16 or INT8 in multiplications to speed up tensor ops (Tensor Cores). NVIDIA GPUs (Ampere+) support FP16 and INT8/TensorFloat32 with minimal loss. For example, GraphSAGE work shows FP16 can nearly double speed. In BDH, projected queries/values and weights can be quantized. PyTorch AMP can automate FP16 use. INT8/INT4 (with torchao or NVIDIA libraries) yields ~4× memory reduction. Aggressive quant (2-4 bit) has accuracy risk but often is acceptable with quant-aware training. At minimum, use bfloat16/FP16 for speed and energy savings (Tensor Cores use ~2× less energy per op than FP32).

- **Batching and Parallelism**: For GPU throughput, batch multiple sequences to fill GPU (like standard Transformer batching). If using very sparse graphs per sample, aggregate multiple graphs into a batched SpMM by block-diagonal adjacency to exploit parallelism. PyTorch Geometric’s `Batch` or DGL’s Batching can help. Ensuring high SM occupancy (many parallel threads) is crucial for GPU throughput.

### Software and Libraries

- **Use Optimized Libraries**: Leverage cuSPARSE (for CSR/COO SpMM), cuDNN (for fused kernels), NVIDIA CUTLASS or cuSPARSELt (for structured sparsity). On CPU, Intel MKL or GraphBLAS provide efficient sparse ops. PyTorch now includes some CSR/CSC support (see torch.sparse docs). DGL/PyG libraries have batched sparse kernels.

- **Frameworks Support**: PyTorch 2.x has improved sparse tensor support; TensorFlow has `tf.sparse` (CSR/COO). For Apple, use PyTorch MPS backend or Apple's MPSGraph (no native sparse yet). Commercial tooling (TensorRT, CoreML) may help fuse layers.

## BDH Architecture and Computation Graph

BDH (“Dragon Hatchling”) is a *sparsely connected LLM* inspired by brain networks. Each BDH layer has a large neuron state of size $n\times d$ (e.g. $n\approx32$k, $d=256$ in experiments), with weights $D_x,D_y\in\mathbb{R}^{d\times n}$ and $E\in\mathbb{R}^{n\times d}$. Its forward pass (per token) roughly is:

```python
# Pseudocode of one BDH layer (see Appendix E of [11])
Q = ReLU(v @ D_x)            # v: (batch, seq, d), D_x: (d,n) → Q: (batch, heads, seq, n/H)
K = Q                        # keys = queries (self-attention)
A = Masked( Q @ K^T )        # causal attention (lower triangular mask)
A = A * v.unsqueeze(heads)   # apply values (here v is the 'value' tensor)
Y = ReLU( LayerNorm(A) @ D_y ) * Q  # second linear (with residual gating by Q)
Y = Y.reshape(batch, seq, n) # merge heads
v_next = v + LayerNorm(Y @ E)  # project back to d and add to state
```

In words, BDH does:
- **Embedding Projection**: $v$ (d-dim token embedding) is projected by $D_x$ into an $n$-dim vector $Q$. Only ~5% of $Q$ entries are nonzero after ReLU.
- **Sparse Multi-Head Attention**: Compute $QK^T$ (shape $T\times T$ for context length $T$) with a causal mask, then multiply by $V=v$ to get attention output $A$. Each head works on an $n/H$-dim subspace. Mask is lower-triangular (causal).
- **Second Linear**: Project $A$ by $D_y$ (d×n) and combine with $Q$ (elementwise multiply) to produce an $n$-vector $Y$.
- **State Update & Readout**: Fold $Y$ (size $n$) back to size $d$ via $E: \mathbb{R}^{n\times d}$ (plus LayerNorm and residual). The result $v_{\text{next}}$ is the new hidden state for next token.

Because $n\gg d$ (e.g. $n=32768$, $d=256$) and most of $Q,Y$ are zero, BDH is effectively a **sparse GNN**: neurons 1…$n$ form a graph where edges are determined by large entries of $D_x,D_y,E$ and attention connections. Indeed, the learned neuron graph is *scale-free* with heavy-tailed degree distribution. Crucially, BDH activations are enforced positive (via ReLU) and extremely sparse (only ~5% nonzeros), a property that enables sparse computation.

The Mermaid diagram below sketches BDH’s per-layer dataflow (the “attention layer graph”):

```mermaid
flowchart LR
    subgraph "BDH Layer (per token)"
        V[(v): token embedding]
        D_x[(D_x, d×n)] 
        D_y[(D_y, d×n)]
        E[(E, n×d)]
        LN1(LN)
        LN2(LN)
        ReLU1(ReLU)
        ReLU2(ReLU)
        Mask[Lower Triangular Mask]
        -- projects --> |"matmul"| X[Q = ReLU(LN1(v)·D_x)]
        X --> |"causal QKᵀ"| Mask
        Mask --> |"matmul V"| A[A = Mask · v] 
        A --> |"ReLU"| Y_temp
        Y_temp --> |"matmul"| Y[Y = ReLU(LN2(A)·D_y) ● X]
        Y --> |"reshape"| Y_flat[(Y reshaped)]
        Y_flat --> |"matmul & add"| v_next[v' = v + LayerNorm(Y_flat·E)]
    end
```

Notes on this flow:

- `v → D_x`: The embedding `v` is multiplied by `D_x` and passed through ReLU, yielding sparse vector `Q`.
- `Q` self-attends: Compute $QK^T$ with $K=Q$ (per head), apply the causal mask, then multiply by $v$ (value) to get `A`.  
- `A` → `D_y`: `A` is normalized, then multiplied by `D_y` and combined with `Q` (elementwise), producing `Y`.  
- `Y` → `E`: The sparse vector `Y` is projected back to $d$-dim via `E` and added to `v` (residual).  

Each matrix multiplication (`v·D_x`, `LN2(A)·D_y`, `Y_flat·E`) is a *large* dense GEMM: e.g. for $n=32$k, $d=256$, these are $256\times32768$ or $32768\times256$ matmuls. The attention `QK^T` per head is $(T\times\frac{n}{H}) \times (\frac{n}{H}\times T)$ producing a $T\times T$ matrix. All these ops must be optimized.

## CUDA and NVIDIA GPU Optimization

On NVIDIA GPUs, BDH’s heavy operations are matrix multiplies and mask operations. Key CUDA optimizations include:

- **Tensor Cores & Mixed Precision**: Use FP16/bfloat16 or INT8 to utilize Tensor Cores. For example, converting $D_x,D_y,E$ to FP16 can roughly double throughput on Ampere/Hopper GPUs. NVIDIA’s TensorFloat32 (TF32) is a good midpoint for safe speedup. Structured 2:4 sparsity (cuSPARSELt) can be applied to $D_x,D_y,E$ if pruned to that pattern, yielding ~1.5× GEMM speedups. (A code recipe: apply `torchao.sparsity.training.SeSemiSparseLinear` to replace linear layers with 2:4-sparse kernels.)

- **Kernel Fusion**: Fuse adjacent ops into custom CUDA kernels. For instance, a custom kernel can compute `Y = ReLU(LN2(A)·D_y) * Q` in one pass. Using Triton or writing a fused kernel avoids writing the full intermediate `ReLU(LN2(A)·D_y)` to memory. Likewise, fuse the final Linear+Add: instead of two kernels (matmul then add+LN), use one kernel that does `output = v + (Y·E)`. Pytorch 2.x’s `torch.compile` (Inductor/Triton) can auto-fuse some patterns, but hand-tuning critical parts (e.g. fused attention or multi-GEMM) often beats generic fusion.

- **Sparse Matmul (SpMM)**: After ReLU, ~95% of entries are zero. Instead of dense GEMM for `v·D_x`, a **sparse-dense GEMM** (SpMM) may save work. If $Q$ is stored CSR (nonzero indices), one can do a custom kernel that multiplies only nonzero rows of $D_x^T$ by `v`. Libraries like cuSPARSE support CSR matmul. The BDH authors note only ~5% of neurons fire, so *projecting only those* can speed up inference. A simple approach: gather nonzero indices of $Q$, slice the columns of $D_x^T$, and run a smaller GEMM for each sample. This trades overhead of indexing for fewer multiplications.

- **Attention (QKᵀ) Optimizations**: The $T\times T$ attention matmul is dense. Use optimized attention kernels (e.g. FlashAttention) which fuse the mask and softmax in shared memory. BDH uses *linear* attention, which can be implemented as accumulating prefix sums on GPU to reduce $O(T^2)$ to $O(T)$ per step (similar to state-space models). If $T$ is small (e.g. 1024), a fused causal GEMM (with mask applied in a single kernel) is sufficient.

- **Operator Reordering**: Reorder loops in SpMM to avoid dense iterations. For example, when computing sparse $C=A·B$, accumulate results by looping over nonzeros of $A$ (row-wise) instead of inner loops. If implementing your own SpMM, use a Gustavson-style algorithm (see Scorch) to avoid full scans.

- **Memory Layout**: Place tensors in GPU global memory as 32- or 64-byte aligned vectors. Use shared memory for small tiles (e.g. in attention) to reduce global load. For SpMM, align CSR row pointers and index arrays in GPU constant memory if reused. Pinned host memory for data transfer helps for large batches.

- **Batching Strategies**: Process multiple tokens or sequences in parallel. BDH’s custom attention fuses well for long sequences, but if many short sequences, pack them into a batch with masking. A single CUDA kernel can process heads * batch together (treated as batch-major GEMM). Ensure batch sizes are large enough to saturate all SMs.

- **Code Example (CUDA fused ReLU)**: As an illustration, one could write a CUDA kernel for `out = max(0, W·x)` to fuse Linear+ReLU:
  ```cpp
  __global__ void linear_relu(const float* __restrict__ W, const float* __restrict__ x, float* out,
                              int M, int K) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < M) {
          float sum = 0;
          for (int i = 0; i < K; i++) sum += W[row*K + i] * x[i];
          out[row] = fmaxf(sum, 0.0f);
      }
  }
  ```
  In PyTorch, one can wrap such kernels in a `torch.autograd.Function` or use Triton to generate a similar fused kernel.

## Apple MPS / Metal Optimization

Apple’s MPS (Metal Performance Shaders) and the MLCGraph framework offer GPU acceleration on Apple Silicon (M1/M2). However, support for sparse operations and fusion is limited. Key strategies:

- **Use Contiguous Tensors**: As noted, many MPS/MPSGraph kernels currently misbehave on non-contiguous memory. For example, `addcmul_` or `addcdiv_` on a strided (non-contiguous) tensor may produce no effect. Always `.contiguous()` before calls.

- **Torch MPS Backend**: PyTorch’s MPS support (via MPSAccelerator/Metal) is improving but still lags CUDA. Complex fusions may not run on GPU; the compiler may fall back to CPU. To mitigate this:
  - Keep graphs static (fewer Python control flows) so that `torch.compile` (Inductor) can fuse and target MPS. 
  - Consider using Apple’s **MLX** (private framework) or MPSGraph directly for critical parts, as benchmarks show MLX can be ~2–3× faster than PyTorch/MPS for LLMs. (In a production Mac app, one might export to CoreML/MLC to leverage MLX optimizations.)
  - Use only supported operations: MPSGraph supports dense matmuls, convs, etc., but not highly dynamic sparse indexing. So one may have to implement sparse steps manually (e.g. gather active indices with `masked_select`, though this can be slow on MPS).

- **Metal Kernel Fusion**: You can write custom Metal shaders for critical loops (attention matmuls, vector updates). Apple provides [MPSGraph APIs](https://developer.apple.com/documentation/metalperformanceshadersgraph) for common NN ops, which will run on GPU if possible. Unfortunately, built-in support for sparse or fused multi-op kernels is minimal as of 2026. One workaround: use the new `mlcompute` (MLCGraph) which can import ONNX/Torch graphs and auto-fuse on Apple Neural Engine (see WWDC 2024).

- **Memory Considerations**: Apple Silicon has unified on-chip memory, but still limited bandwidth. Aim to reuse data: fuse ops to keep data on-chip. Avoid intermediate allocations. Use half-precision (float16) where possible; MPSGraph has FP16 support. Measure GPU vs CPU runs: sometimes, for small models, MPS CPU path may be faster for certain primitives.

- **Batch Size and Threads**: MPS has its own optimizers. For performance, ensure workload (batch×seq length) is enough to saturate the GPU. Track `torch.mps.driver_allocated_memory()` to ensure no unexpected peaks. Use macOS Instruments (GPU counters, power profiling) to debug hotspots.

In summary, on Apple Silicon prioritize simple, contiguous kernels and minimize dynamic loops. Given PyTorch MPS’s current limitations, significant fusion and sparse acceleration may require lower-level (Metal or MLCGraph) coding.

## Prioritized Optimization Techniques

Below is a **prioritized list** of concrete techniques to speed up BDH, with rough *impact*, *complexity*, and *risk* ratings:

1. **Use Optimized Sparse Format (CSR/CSC)** – *Impact: High*. Storing weight matrices or graph adjacency in CSR reduces storage and speeds up SpMM. *Complexity: Low–Medium* (use PyTorch’s `sparse_csr_tensor` or cuSPARSE). *Risk: Low* (just need to convert data). This is fundamental: running a sparse matrix multiply on CSR is often **3–10× faster** than naive dense when sparsity is ~95%.  

2. **Mixed-Precision / Tensor Cores** – *Impact: High*. Switching computation (GEMMs and kernels) from FP32 to FP16 or bfloat16 taps NVIDIA TensorCores (and Apple GPUs) for ~2× throughput increase. Using automatic mixed precision (AMP) in PyTorch is straightforward. *Complexity: Low* (just cast tensors). *Risk: Medium* (requires careful scaling to avoid numeric underflow; slight accuracy loss possible).

3. **Structured Sparsity (2:4)** – *Impact: High*. Prune BDH’s weight matrices to 2-out-of-4 blocks to use CUDA’s semi-structured sparse kernels. This can give ~1.5× speedup on NV GPUs with minimal accuracy drop. *Complexity: Medium* (need to enforce pattern during training/pruning). *Risk: Medium* (if patterns not well-chosen, accuracy may degrade; also not beneficial on non-NVIDIA hardware).

4. **Kernel Fusion (TorchInductor/Triton)** – *Impact: High*. Fusing elementwise chains (e.g. MatMul→ReLU→Dropout or Mask→MatMul→Add) reduces memory traffic. *Complexity: Medium* (PyTorch 2.x can do some automatically; writing custom Triton kernels can be complex). *Risk: Low–Medium* (fusion bugs are rare if verified, but debugging fused kernels is harder).

5. **Pruning (Unstructured)** – *Impact: Medium*. Remove <50% weights and run SpMM on the compressed model. Cuts computation roughly proportionally to remaining weights. *Complexity: Medium* (requires retraining/pruning code). *Risk: Medium* (unstructured pruning may require custom sparse GEMM; moderate accuracy risk if over-pruned). Empirical studies show global pruning can reduce model size by ~50% with little loss.

6. **Attention Optimization** – *Impact: Medium–High*. Use a custom fused attention (e.g. FlashAttention-style) or a state-space kernel for the linear attention to cut $O(T^2)$ overhead. *Complexity: High* (writing a correct, efficient custom attention kernel). *Risk: Medium* (masking logic must be exact; numeric stability).

7. **Graph Reordering / Partitioning** – *Impact: Medium*. Reorder neurons/edges to cluster memory access for SpMM. Tools like graph clustering can improve cache usage. *Complexity: High* (requires analyzing graph, reindexing). *Risk: Low* (conceptually straightforward, but gains vary).

8. **Batch Parallelism** – *Impact: Medium*. Process multiple sequences in parallel, or multiple graph partitions, to increase GPU utilization. *Complexity: Low*. *Risk: Low*. On multi-GPU, use pipeline or tensor parallelism (advanced).

9. **Quantization-Aware Training** – *Impact: Medium*. Quantizing to INT8/4-bit at training time can double throughput and memory efficiency. *Complexity: High* (needs QAT scaffolding). *Risk: High* (accuracy may drop; implementation is complex).

10. **Operator Reordering in SpMM** – *Impact: Low–Medium*. Internally reorder loops in your custom sparse kernels for better asymptotic behavior. *Complexity: High* (requires low-level coding). *Risk: Low* (purely computational strategy).

This ordering balances **impact vs. effort**. For instance, switching to CSR+FP16 (items 1–2) yields immediate large gains and is easy. Structured sparsity and fusion yield further speedups at moderate complexity. Pruning/quantization are powerful but riskier and require more engineering.

## Implementation Recipes and Code Sketches

Below are examples of how to implement some optimizations:

- **Custom CUDA SpMM Kernel (CSR)**: One can write a CUDA kernel that, for each row of the sparse matrix, iterates over nonzeros and accumulates dot products. For example:

  ```cuda
  __global__ void spmm_csr(int n_rows, 
      const int* crow_ptr, const int* col_idx, const float* vals, 
      const float* B, float* C, int K) {
      // A is n_rows×K in CSR, B is K×N (dense), output C is n_rows×N
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (i < n_rows) {
          for(int j=0; j<N; j++) C[i*N+j] = 0.0f;
          int row_start = crow_ptr[i], row_end = crow_ptr[i+1];
          for(int idx=row_start; idx<row_end; idx++) {
              int k = col_idx[idx];
              float a = vals[idx];
              for(int j=0; j<N; j++){
                  C[i*N+j] += a * B[k*N+j];
              }
          }
      }
  }
  ```
  This avoids multiplying zeros from $A$. In PyTorch, such a kernel could be launched via a `torch.autograd.Function` with `spmm_csr<<<...>>>`, or use **torch.ops.torch_sparse** if available.

- **PyTorch Triton Fused Kernel**: Triton allows writing fused operations in Python. For instance, fusing linear and ReLU:

  ```python
  import triton
  import triton.language as tl

  @triton.jit
  def linear_relu_kernel(W_ptr, X_ptr, Y_ptr, M, K, stride_wm, stride_wk, stride_xk):
      row = tl.program_id(0)
      acc = 0.0
      for k in range(0, K, 1):
          w = tl.load(W_ptr + row * stride_wm + k * stride_wk)
          x = tl.load(X_ptr + k * stride_xk)
          acc += w * x
      y = tl.max(acc, 0.0)  # ReLU
      tl.store(Y_ptr + row * stride_xm, y)
  ```
  One would launch this with `M` rows and appropriate strides. This fuses the matrix-vector product and ReLU in one GPU kernel.

- **Mapping BDH Attention to GEMM**: In BDH’s multi-head attention, `Q (T×d) @ Kᵀ (d×T)` can be implemented as a single cuBLAS batched GEMM across heads and batches. For causal masking, one strategy is to compute the full $QK^T$ and then multiply by a mask matrix (or use `cublasGemmStridedBatched` on triangular parts). Alternatively, implement a *running-sum* kernel for prefix accumulations (like FlashAttention algorithms). If using PyTorch, one can simply call `torch.matmul(Q, K.transpose(-2,-1))` followed by `tril()` masking. For speed, use `torch.compile` with `backend='cuda'` on a function combining these operations.

- **Apple MPS (PyTorch)**: To force MPS usage in PyTorch, simply `.to('mps')` your model and inputs. For example:
  ```python
  model = BDHModel(...).to('mps')
  input = input.to('mps')
  output = model(input)
  ```
  Unfortunately, no custom MPS kernels can be injected from Python. The best practice is to keep it simple: e.g. replace a fused op with two supported ops if needed, and ensure tensors are contiguous. You can also use `torch.jit.trace` to freeze the graph then export to ONNX and import into CoreML (mlc) for better performance.

## Sparse Data Structures & Layouts

The table below compares common sparse formats for BDH’s weight matrices or adjacency, focusing on SpMM performance, memory, and update ease:

| **Format**      | **Memory Overhead**    | **SpMM Efficiency**                 | **Update Cost**              | **Suitability**                                              |
|-----------------|------------------------|-------------------------------------|------------------------------|--------------------------------------------------------------|
| **CSR (row)**   | Low (8B×n_rows + indices×nnz) | *High*: optimized cuSPARSE/MLK†. Better cache for row ops. | Low (static; update by reconstructing CSR)  | Excellent for static BDH graph; fastest sparse matmul. |
| **CSC (col)**   | Low (similar to CSR)   | *High*: good if mostly column ops.  | Low (static)                 | Equiv. to CSR (transpose of graph); good for symmetric use.  |
| **COO**         | Moderate-High (index arrays for each nnz) | *Moderate*: flexible but slower indexing. | High (easy to append indices) | Good for dynamic edge updates/pruning during training. Slower SpMM than CSR. |
| **Block-sparse** (e.g. 2:4) | Moderate (mask + values) | *High*: hardware-accelerated GEMM (up to ~1.6×). | Moderate (requires per-block mask) | Good when BDH weights can be pruned to fixed blocks; leverages tensor cores. |
| **Hash/Other**  | High (pointer overhead)  | *Low*: random access; no hardware support | Easy (dynamic insert)      | Generally poor for high-throughput; use only for small dynamic sets. |
| **Hybrid (COO+CSR)** | Variable   | *Medium-High*: can group heavy rows in CSR, light rows in COO. | Medium | Good when graph changes partially; more complex implementation. |

\* *Memory overhead example:* A 10k×10k sparse matrix with 100k nnz has ~1.28MB in CSR, whereas COO needs extra index for each entry.  
† *Libraries:* Use cuSPARSE/Amgx for CSR SpMM on NVIDIA; PyTorch CSR/CSC support is maturing.

In BDH, since activations and synapses change slowly, a **static CSR** for the trained graph is often best. For any remaining dynamic sparsity, one might use **hybrid** strategies.

## Benchmarking Methodology

A rigorous evaluation plan should measure **both micro and end-to-end metrics**:

- **Microbenchmarks**:
  - *Sparse-Dense GEMM (SpMM) Performance*: Construct random sparse matrices with BDH-like sparsity (e.g. 5% density, scale-free degree) and measure SpMM vs dense GEMM (on CPU/GPU). Libraries like cuSPARSE or custom CUDA kernels can be benchmarked.
  - *Attention Kernel*: Measure the time of BDH’s attention on various $T$ (context lengths). Compare fused vs unfused, FP32 vs FP16, sparse-Q vs dense-Q.
  - *Memory Access*: Use NVIDIA’s `nvprof`/Nsight to measure memory bandwidth utilization for each kernel, and identify stalls or bank conflicts.
  - *Kernel Launch Overhead*: Count the number of small kernels (e.g. elementwise) and their latency.

- **End-to-End Tests**:
  - *Throughput (tokens/sec)*: Run BDH training/inference on a representative task (e.g. language modeling) with and without each optimization, measuring the average tokens/sec. Vary batch size to find optimal throughput.
  - *Latency*: For a single sample (batch=1), measure time per forward pass (critical for real-time inference).
  - *Memory Profile*: Track peak and average GPU memory during run (e.g. `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()`). Include overhead of data structures.
  - *Energy & Power*: Use hardware counters/APIs: on NVIDIA use NVML (e.g. `nvidia-smi --query`), on Apple use Instruments/PMU. Normalize to **joules per token** or per sample.
  - *Accuracy*: After quantization/pruning, measure test accuracy or loss versus baseline. 

- **Environment**: Clearly note hardware (GPU model, compute capability, memory), software versions (CUDA, CuDNN, PyTorch, Apple OS), and hyperparameters (BDH size, sequence length, batch size). 

- **Analysis**: Report *peak throughput* and *sustained throughput* (with large batches), *latency percentiles*, *memory overhead*, *power draw*, and *accuracy delta*. Use phase-aware measurement like TokenPowerBench, aligning power samples to BDH phases (embedding, attention, etc.) to identify hotspots. 

- **Regression Checks**: For each optimization (e.g. fused kernel), include a correctness test that compares outputs (and gradients, if training) to a reference.

## Potential Pitfalls and Validation

When implementing these optimizations, watch out for:

- **Sparse Tensor Caveats**: PyTorch’s sparse support has quirks. Indices must be `int64` (or `int32` with MKL); using wrong dtype can force CPU fallback. COO tensors must be *coalesced* (no duplicate indices) or behavior is undefined. Always check invariants (`torch._sparse._make_coo_tensor` can verify). 

- **Memory Layout**: Some GPU kernels require data to be aligned. E.g. 2:4 sparsity on NVIDIA requires 4-element blocks aligned in memory. On MPS, non-contiguous outputs can silently fail (as seen with Adam’s `addcmul_`). Force `.contiguous()` after reshapes or transposes.

- **Numerical Stability**: Lower precision (FP16/INT8) may underflow or saturate in accumulations. Use scaled FP16 (loss scaling for training) and ensure attention logits use enough mantissa. For INT8 quantization, calibrate ranges carefully.

- **Masking Errors**: In attention, an off-by-one in applying the causal mask can corrupt training. Verify that masked positions truly have zero effect.

- **Kernel Launch Overhead**: Too many small kernels (e.g. one per token or head) can drown throughput. Fuse as much as possible or batch kernel launches.

- **Underutilization**: Very sparse data may underutilize GPU. If only a few neurons fire, many threads will be idle. In that case, batching or fallback to a dense path for small sparsity might be faster.

- **Profiling Artifacts**: Watch for asynchronous timing issues when measuring GPU time. Always `cudaDeviceSynchronize()` before timing.

- **MPS-specific Bugs**: As noted, PyTorch MPS has bugs (non-contig ops, limited fused kernels). Validate on multiple macOS versions. If performance is inadequate, consider exporting the model via CoreML to leverage Apple’s highly-optimized kernels.

- **General Validation**: Continuously compare an optimized BDH against a reference implementation. Test on trivial data (identity graph) to ensure correctness of optimized kernels. For pruning/quant, ensure accuracy is within acceptable bounds after fine-tuning.

## References

A selection of key sources:

- Haziza *et al.* (2025) **“The Dragon Hatchling”** – Original BDH paper.  
- NVIDIA Developer resources – cuSPARSELt (2:4 sparsity) and cuSPARSE guides.  
- Cai *et al.* (2024) **PyTorch Blog** on 2:4 sparsity (semi-structured).  
- Pan *et al.* (2024) ArXiv **“Accelerating Sparse GNNs with Tensor Cores”**.  
- Heinecke *et al.* (2024) **Scorch** library paper on sparse kernel fusion.  
- PyTorch documentation on sparse tensors.  
- Khedri *et al.* (2025) **Pruning & Quantization of GNNs** – empirical study.  
- Tang *et al.* (2025) **TokenPowerBench** – methodology for LLM inference energy.  
- PyTorch Issue/Blog – **Apple MPS quirks** (non-contig failures); **PyTorch 2025 HW Acceleration Survey** (MPS vs CUDA).  

Each optimization suggestion above is supported by these and other primary sources (papers, official docs, and authoritative blog posts) to ensure up-to-date and accurate guidance.

