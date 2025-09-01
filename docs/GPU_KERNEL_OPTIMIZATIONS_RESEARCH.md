# Research: GPU Kernel Optimizations for LLM Inference

## Summary
This research covers four critical GPU kernel optimizations that have revolutionized LLM inference performance: FlashAttention (reducing memory movement through tiling and online softmax), GEMM optimization (leveraging tensor cores and advanced tiling strategies), Quantization kernels (enabling efficient low-precision computation), and PagedAttention (managing non-contiguous memory for efficient KV cache). These techniques collectively enable 2-10x performance improvements in modern LLM serving systems.

## Authoritative Sources
- [FlashAttention Paper (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135): Original FlashAttention algorithm and IO-aware optimization
- [FlashAttention-2 Blog (Stanford CRFM)](https://crfm.stanford.edu/2023/07/17/flash2.html): Improvements in parallelism and performance
- [FlashAttention-3 (Tri Dao)](https://tridao.me/blog/2024/flash3/): Asynchronous execution and FP8 support
- [CUDA Matrix Multiplication Optimization (Sibo Ehm)](https://siboehm.com/articles/22/CUDA-MMM): Detailed GEMM kernel optimization techniques
- [NVIDIA WMMA Documentation](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/): Tensor Core programming interface
- [vLLM PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html): Memory management for LLM serving
- [PagedAttention Paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180): Efficient memory management algorithm

## 1. FlashAttention: IO-Aware Exact Attention

### Mathematical Foundations
FlashAttention computes exact attention: `Attention(Q, K, V) = softmax(QK^T/√d_k)V`

The key innovation is computing this without materializing the full N×N attention matrix in GPU high-bandwidth memory (HBM).

### Algorithm Details

#### Memory Hierarchy Exploitation
- **GPU Memory Architecture**: A100 has 40-80GB HBM (1.5-2.0 TB/s bandwidth) vs 192KB SRAM per SM (19 TB/s bandwidth)
- **IO-Awareness**: Minimizes data movement between HBM and SRAM by keeping intermediate results in fast SRAM

#### Tiling Strategy
```
1. Divide Q, K, V into blocks of size B×d (B = block size, d = embedding dimension)
2. Load blocks from HBM to SRAM
3. Compute attention for block in SRAM
4. Update output in HBM incrementally
```

**Block Size Calculation Example**:
- For SRAM size M=1000, d=5: Block size = 1000/(4×5) = 50 vectors
- Process 50 q, k, v, o vectors at a time to minimize HBM reads/writes

#### Online Softmax Computation
The critical innovation enabling block-wise processing:

```python
# Pseudo-code for online softmax with running statistics
for block in blocks:
    # Compute scores for current block
    S_block = Q_block @ K_block.T / sqrt(d)
    
    # Update running max (for numerical stability)
    m_new = max(m_old, max(S_block))
    
    # Update running sum with correction factor
    l_new = exp(m_old - m_new) * l_old + sum(exp(S_block - m_new))
    
    # Accumulate output with rescaling
    O_new = exp(m_old - m_new) * O_old + exp(S_block - m_new) @ V_block
```

This maintains correctness while processing attention in tiles.

#### Kernel Fusion
- Keeps intermediate values in SRAM throughout computation
- Fuses softmax, dropout, and masking operations
- Writes only final result back to HBM

### Performance Characteristics
- **Memory Complexity**: Reduced from O(N²) to O(N) HBM accesses
- **Speed**: 3× faster on GPT-2 (seq length 1K), 15% speedup on BERT-large
- **Memory Usage**: Linear in sequence length instead of quadratic
- **GPU Utilization**: 35% on H100 (FlashAttention-2), improved to 75% (FlashAttention-3)

### FlashAttention-2 Improvements
- 2× faster than FlashAttention-1
- Reaches 230 TFLOPs/s on A100 (FP16/BF16)
- Reduced rescaling operations
- Optimized bound-checking and causal masking

### FlashAttention-3 Innovations
1. **Asynchronous Execution**: Warp specialization overlaps computation and data movement
2. **Interleaved Operations**: Overlaps low-throughput softmax with high-throughput GEMM
3. **FP8 Support**: Nearly doubles throughput with block quantization
4. **Performance**: 740 TFLOPs/s FP16, 1.2 PFLOPS FP8 on H100

## 2. GEMM (General Matrix Multiplication) Optimization

### Core Optimization Strategies

#### Memory Access Optimization
```cuda
// Coalesced memory access pattern
// Each thread in warp accesses consecutive memory addresses
float4 data = reinterpret_cast<float4*>(global_ptr)[thread_id];
```

#### Hierarchical Tiling
1. **Block Tiling**: Cache matrix chunks in shared memory (48KB on modern GPUs)
2. **Warp Tiling**: Divide work across warp schedulers (4 per SM)
3. **Thread Tiling**: Each thread computes multiple output elements
4. **Vectorized Access**: Use float4 loads for 4× memory bandwidth

#### Implementation Evolution
```
Naive implementation:        1.3% of cuBLAS performance
+ Global memory coalescing:   8.5% performance
+ Shared memory caching:     12.8% performance  
+ Block/warp tiling:         93.7% performance
```

### Tensor Core Utilization

#### WMMA API Usage
```cuda
// Tensor Core matrix multiply-accumulate
wmma::fragment<wmma::matrix_a, M, N, K, half> a_frag;
wmma::fragment<wmma::matrix_b, M, N, K, half> b_frag;
wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

// Load data from shared memory to fragments
wmma::load_matrix_sync(a_frag, a_shared, lda);
wmma::load_matrix_sync(b_frag, b_shared, ldb);

// Perform matrix multiply-accumulate
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

// Store result back to shared/global memory
wmma::store_matrix_sync(c_global, c_frag, ldc);
```

#### Supported Configurations
- **Matrix Sizes**: m16n16k16 (Volta/Turing), m16n8k16, m8n32k16 (Ampere+)
- **Data Types**: FP16 input → FP32 accumulation, INT8 input → INT32 accumulation
- **Alignment**: Best performance with dimensions as multiples of 16 (128 on A100)

### Advanced Optimization Techniques

#### Double Buffering
```cuda
// Prefetch next tile while computing current tile
__shared__ float A_shared[2][TILE_SIZE];
int buffer = 0;

// Load first tile
load_tile(A_shared[0], ...);
__syncthreads();

for (int tile = 0; tile < num_tiles; tile++) {
    // Prefetch next tile
    if (tile < num_tiles - 1) {
        load_tile(A_shared[1 - buffer], ...);
    }
    
    // Compute with current tile
    compute_tile(A_shared[buffer], ...);
    
    __syncthreads();
    buffer = 1 - buffer;
}
```

#### Bank Conflict Elimination
- Use padding for shared memory arrays
- Permuted access patterns for MMA instructions
- Swizzle addressing for improved L2 cache hit rates

### Performance Benchmarks
- **FP32 GEMM**: 20.16 TFLOPS on RTX 3090 (80-90% of cuBLAS)
- **FP16 with Tensor Cores**: 2.5× speedup over FP32
- **INT8 with Tensor Cores**: 4× speedup over FP32
- **Optimization Impact**: Up to 95% of cuBLAS performance achieved

## 3. Quantization Kernels

### Quantization Methods

#### INT8 Quantization
```python
# Symmetric quantization
scale = max(abs(tensor.max()), abs(tensor.min())) / 127
quantized = round(tensor / scale).clip(-128, 127).astype(int8)

# Asymmetric quantization with zero point
scale = (tensor.max() - tensor.min()) / 255
zero_point = round(-tensor.min() / scale)
quantized = round(tensor / scale + zero_point).clip(0, 255).astype(uint8)
```

#### INT4 Quantization
- 4-bit representation (16 possible values)
- Typically uses group quantization (e.g., 128 elements share scale)
- Requires specialized kernels for efficient unpacking

#### FP8 Formats (E4M3, E5M2)
- **E4M3**: 1 sign, 4 exponent, 3 mantissa bits (better precision)
- **E5M2**: 1 sign, 5 exponent, 2 mantissa bits (better range)
- Hardware support on H100 GPUs

### Advanced Quantization Algorithms

#### GPTQ (Generalized Post-Training Quantization)
```python
# GPTQ algorithm pseudo-code
for layer in model.layers:
    W = layer.weight  # Original FP16 weights
    H = compute_hessian(W, calibration_data)  # Second-order information
    
    for column in range(W.shape[1]):
        # Quantize column independently
        w_quant = quantize_optimal(W[:, column], H)
        
        # Update remaining columns to compensate
        error = W[:, column] - dequantize(w_quant)
        W[:, column+1:] -= outer_product(error, H_inv[column])
```

**Characteristics**:
- Mixed INT4/FP16 precision
- Row-wise quantization minimizing reconstruction error
- 4× memory savings with on-the-fly dequantization

#### AWQ (Activation-aware Weight Quantization)
```python
# AWQ algorithm concept
# 1. Collect activation statistics
activations = collect_activations(model, calibration_data)

# 2. Identify salient weights (top 1%)
importance = compute_weight_importance(weights, activations)
salient_mask = importance > threshold

# 3. Keep salient weights in FP16, quantize others to INT4
weights_mixed = {
    'fp16': weights[salient_mask],
    'int4': quantize(weights[~salient_mask])
}
```

**Performance**: AWQ 4-bit ~1.2× faster than GPTQ 4-bit

### On-the-fly Dequantization

#### Fused Kernel Implementation
```cuda
__global__ void gemm_with_dequant(
    const int4* W_quantized,    // Packed INT4 weights
    const float* scales,         // Per-group scales
    const float* X,              // FP16 activations
    float* Y                     // Output
) {
    // Load quantized weights
    int4 w_packed = W_quantized[...];
    
    // Unpack and dequantize in registers
    float4 w_float;
    w_float.x = (w_packed.x & 0xF) * scales[group_id];
    w_float.y = ((w_packed.x >> 4) & 0xF) * scales[group_id];
    // ... continue unpacking
    
    // Perform computation with dequantized values
    float acc = dot(w_float, x_tile);
}
```

### Calibration and Scaling

#### Calibration Dataset Requirements
- Typically 100-1000 samples from training distribution
- Used to compute activation statistics and optimal scales
- Critical for maintaining model accuracy

#### Dynamic vs Static Quantization
- **Static**: Scales computed offline, fixed during inference
- **Dynamic**: Scales computed per-batch, better accuracy but slower

### Hardware-Specific Optimizations

#### Marlin Kernel (A100 Optimized)
- 4-bit only CUDA kernel for Ampere architecture
- Highly parallelized loading, dequantization, and execution
- Optimized for specific tile sizes and warp configurations

#### Performance Impact
- **Memory Reduction**: 75% for INT4, 50% for INT8
- **Throughput**: 2-2.5× increase for memory-bound operations
- **Accuracy Trade-off**: <1% degradation with proper calibration

## 4. Paged Attention Kernels

### Core Architecture

#### Virtual Memory-Inspired Design
```python
# PagedAttention block management
class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=1000):
        self.block_size = block_size
        self.physical_blocks = [Block() for _ in range(num_blocks)]
        self.block_tables = {}  # seq_id -> list of block indices
        
    def allocate_block(self, seq_id):
        free_block = self.find_free_block()
        self.block_tables[seq_id].append(free_block)
        return free_block
```

#### Non-Contiguous Memory Storage
Instead of storing KV cache as contiguous arrays:
```
Traditional: [K_seq1_all_tokens][K_seq2_all_tokens]...
PagedAttention: [Block0][Block7][Block3]...  # Blocks can be anywhere
```

### Kernel Implementation Details

#### Block Table Lookup
```cuda
__global__ void paged_attention_kernel(
    const float* Q,           // Query tensor
    const float* K_blocks,    // Non-contiguous K blocks
    const float* V_blocks,    // Non-contiguous V blocks  
    const int* block_table,   // Logical → Physical mapping
    float* output
) {
    int seq_pos = threadIdx.x + blockIdx.x * blockDim.x;
    int logical_block = seq_pos / BLOCK_SIZE;
    int block_offset = seq_pos % BLOCK_SIZE;
    
    // Indirect addressing through block table
    int physical_block = block_table[logical_block];
    
    // Gather K, V from non-contiguous memory
    float k = K_blocks[physical_block * BLOCK_SIZE + block_offset];
    float v = V_blocks[physical_block * BLOCK_SIZE + block_offset];
    
    // Compute attention with gathered values
    // ...
}
```

#### Gather/Scatter Operations
1. **Gather Phase**: Collect KV values from scattered blocks using block table
2. **Compute Phase**: Perform attention computation with gathered data
3. **Scatter Phase**: Write results back to appropriate memory locations

### Memory Management Strategies

#### Dynamic Block Allocation
```python
def generate_token(prompt, kv_cache):
    # Allocate new block only when current block is full
    if current_block_full():
        new_block = allocate_physical_block()
        update_block_table(seq_id, new_block)
    
    # Write new KV to current block
    write_kv_to_block(new_k, new_v, current_block)
```

#### Memory Sharing with Copy-on-Write
```python
class SharedKVCache:
    def __init__(self):
        self.ref_counts = {}  # block_id -> reference count
        
    def share_blocks(self, src_seq, dst_seq):
        # Share blocks between sequences
        for block_id in self.block_tables[src_seq]:
            self.ref_counts[block_id] += 1
            self.block_tables[dst_seq].append(block_id)
    
    def copy_on_write(self, seq_id, block_id):
        if self.ref_counts[block_id] > 1:
            # Create new block and copy data
            new_block = self.allocate_block()
            copy_block_data(block_id, new_block)
            self.update_block_table(seq_id, new_block)
```

### Integration with FlashAttention

#### Modified Tiling for Paged Memory
```cuda
// FlashAttention with PagedAttention
for (int block = 0; block < num_blocks; block++) {
    // Load block indices from block table
    int k_block_idx = k_block_table[block];
    int v_block_idx = v_block_table[block];
    
    // Load K, V tiles from non-contiguous blocks
    load_tile_indirect(K_tile, K_blocks, k_block_idx);
    load_tile_indirect(V_tile, V_blocks, v_block_idx);
    
    // Standard FlashAttention computation
    compute_attention_tile(Q_tile, K_tile, V_tile, O_tile);
}
```

### Performance Characteristics

#### Memory Efficiency
- **Waste**: <4% (only in last partially-filled block)
- **Traditional**: Up to 60-80% waste from pre-allocation
- **Sharing**: Enables prefix caching, reduces memory by 55% for shared prompts

#### Throughput Impact
- **vLLM Performance**: 2-4× higher throughput than baseline
- **Kernel Overhead**: ~20% compared to contiguous memory
- **Overall System**: 24× improvement over HuggingFace Transformers

### Advanced Features

#### Block-Sparse Patterns
```python
# Sparse attention with PagedAttention
def sparse_paged_attention(Q, block_table, sparsity_mask):
    output = zeros_like(Q)
    
    for block_idx in get_sparse_blocks(sparsity_mask):
        if should_attend(block_idx):
            physical_block = block_table[block_idx]
            block_output = compute_attention_block(Q, physical_block)
            accumulate_output(output, block_output)
    
    return output
```

#### Texture Memory Usage (GPU-specific)
- Potential for texture memory caching of frequently accessed blocks
- Hardware-accelerated interpolation for certain access patterns
- Reduced cache pollution for random access patterns

## Recommended Approach

For implementing high-performance LLM inference, combine these optimizations:

1. **Use FlashAttention-3** for attention layers with FP8 on H100 or FlashAttention-2 with FP16 on older GPUs
2. **Implement PagedAttention** for KV cache management in serving scenarios
3. **Apply quantization** (AWQ or GPTQ) for memory-constrained deployments
4. **Optimize GEMM kernels** using CUTLASS templates or cuBLAS for non-attention operations

## Code Examples

### Complete FlashAttention Forward Pass (Triton)
```python
@triton.jit
def flash_attn_forward(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)
    off_h = tl.program_id(2)
    
    # Initialize accumulator and normalization
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    
    # Iterate through K, V blocks
    for start_n in range(0, N, BLOCK_N):
        # Load Q, K, V tiles
        q = tl.load(Q + offsets_q)
        k = tl.load(K + offsets_k)
        v = tl.load(V + offsets_v)
        
        # Compute scores
        scores = tl.dot(q, tl.trans(k)) / math.sqrt(K)
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp(m_i - m_i_new)
        l_i = l_i * alpha + tl.sum(tl.exp(scores - m_i_new[:, None]), 1)
        
        # Update accumulator
        acc = acc * alpha[:, None] + tl.dot(tl.exp(scores - m_i_new[:, None]), v)
        m_i = m_i_new
    
    # Write output
    acc = acc / l_i[:, None]
    tl.store(Out + offsets_out, acc)
```

### Optimized GEMM with Tensor Cores (CUDA)
```cuda
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void gemm_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Shared memory for tiles
    __shared__ half A_shared[BLOCK_M][BLOCK_K];
    __shared__ half B_shared[BLOCK_K][BLOCK_N];
    
    // Main loop over K dimension
    for (int k = 0; k < K; k += BLOCK_K) {
        // Collaborative loading into shared memory
        load_tile_collaborative(A_shared, A, ...);
        load_tile_collaborative(B_shared, B, ...);
        __syncthreads();
        
        // Warp-level GEMM using Tensor Cores
        for (int kk = 0; kk < BLOCK_K; kk += 16) {
            wmma::load_matrix_sync(a_frag, &A_shared[warp_m][kk], BLOCK_K);
            wmma::load_matrix_sync(b_frag, &B_shared[kk][warp_n], BLOCK_N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    
    // Store result
    wmma::store_matrix_sync(&C[...], c_frag, N, wmma::mem_row_major);
}
```

## Additional Considerations

### Hardware Generation Compatibility
- **Volta (V100)**: FlashAttention-1, basic Tensor Cores
- **Ampere (A100)**: FlashAttention-2, improved Tensor Cores, sparse operations
- **Hopper (H100)**: FlashAttention-3, FP8 support, TMA, warp specialization

### Memory Bandwidth Optimization
- Ensure data alignment (128-byte boundaries on modern GPUs)
- Use vectorized loads (float4, int4) whenever possible
- Minimize bank conflicts in shared memory
- Leverage L2 cache persistence policies

### Debugging and Profiling
- Use Nsight Compute for kernel profiling
- Monitor memory throughput vs theoretical peak
- Check warp occupancy and register pressure
- Validate numerical accuracy with reference implementations

### Production Deployment
- Consider using established libraries (vLLM, TensorRT-LLM) that implement these optimizations
- Profile on target hardware before custom kernel development
- Implement fallback paths for unsupported hardware
- Maintain separate code paths for different precision levels

These GPU kernel optimizations represent the state-of-the-art in accelerating LLM inference, with ongoing research continuing to push the boundaries of what's possible on modern GPU hardware.