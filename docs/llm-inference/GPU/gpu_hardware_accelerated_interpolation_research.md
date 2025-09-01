# Research: GPU Architecture Terminology

## Summary
GPU architecture follows a hierarchical design with specialized processing units (CUDA cores, Tensor cores), a complex memory hierarchy (registers → shared memory → L1/L2 cache → global memory), and a parallel execution model based on warps, blocks, and grids. Understanding the interplay between compute resources, memory bandwidth, and occupancy is crucial for GPU programming efficiency.

## Authoritative Sources
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html): Official comprehensive guide covering all CUDA concepts
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth): Latest H100 architecture details
- [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/): Clear explanation of core programming abstractions
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/): Memory optimization techniques
- [How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels): Memory coalescing patterns
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html): Performance profiling metrics
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html): Optimization strategies and patterns

## Core GPU Concepts (20% that explains 80%)

### 1. Processing Units

#### **CUDA Cores**
- Basic processing units for general-purpose parallel computing
- Execute one floating-point or integer operation per clock cycle
- Modern GPUs contain thousands (e.g., H100 has 18,432 FP32 CUDA cores)
- Grouped into Streaming Multiprocessors (SMs)

#### **Tensor Cores**
- Specialized units for matrix multiplication and AI workloads
- Execute mixed-precision matrix operations (FP8, FP16, BF16, TF32, FP64, INT8)
- 4th generation in H100: up to 6x faster than A100
- Critical for deep learning: 2000-4000 TFLOPS in FP8 vs 60 TFLOPS FP32 on CUDA cores

#### **Streaming Multiprocessor (SM)**
- Independent processing unit containing CUDA cores, Tensor cores, memory, and schedulers
- Executes thread blocks independently
- Contains 128 CUDA cores (H100) or 64 CUDA cores (older architectures)
- Has own L1 cache, shared memory, and warp schedulers
- Maximum 64 warps (2048 threads) active per SM

### 2. Memory Hierarchy

#### **Registers** (Fastest, ~1 cycle)
- Private per-thread storage
- Typically 255 registers per thread
- Zero-overhead access when available
- Spills to local memory if exceeded

#### **Shared Memory** (~30 cycles)
- Per-block storage (48-164 KB per SM)
- 100x faster than global memory
- Divided into 32 banks for parallel access
- Bank conflicts serialize access (avoid with padding)

#### **L1 Cache** (~30 cycles)
- Per-SM cache (128 KB combined with shared memory)
- Caches local and global memory accesses
- Configurable split with shared memory

#### **L2 Cache** (~200 cycles)
- Shared across all SMs
- 50 MB in H100, 40 MB in A100
- Reduces global memory traffic

#### **Global Memory (HBM)** (~500 cycles)
- Main GPU memory (80 GB HBM3 in H100)
- 3 TB/sec bandwidth in H100 (2x improvement over A100)
- Accessible by all threads
- Coalesced access patterns critical for performance

#### **Constant Memory**
- 64 KB read-only memory
- Cached and broadcast to threads
- Optimized for uniform access across warps

#### **Texture Memory**
- Read-only memory with spatial caching
- Hardware interpolation support
- Optimized for 2D/3D spatial locality

### 3. Execution Model

#### **Thread**
- Smallest execution unit
- Has unique ID (threadIdx.x/y/z)
- Executes kernel code
- Private registers and local memory

#### **Warp**
- 32 threads executing in SIMT (Single Instruction, Multiple Thread) fashion
- Basic scheduling unit
- All threads in warp execute same instruction
- Divergence causes serialization

#### **Block**
- Group of threads (max 1024)
- Executes on single SM
- Threads can synchronize via `__syncthreads()`
- Share memory within block
- Has unique ID (blockIdx.x/y/z)

#### **Grid**
- Collection of blocks
- Represents entire kernel launch
- Can be 1D, 2D, or 3D
- Size determined by problem domain

#### **Kernel**
- Function executed on GPU marked with `__global__`
- Launched from host (CPU)
- Runs N times in parallel by N threads
- Syntax: `kernel<<<blocks, threads>>>(args)`

#### **Thread Block Clusters** (Compute Capability 9.0+)
- Groups of thread blocks
- Co-scheduled on GPU Processing Cluster (GPC)
- Maximum 8 blocks per cluster
- Enable direct SM-to-SM communication

### 4. Performance Metrics

#### **Occupancy**
- Ratio of active warps to maximum possible warps per SM
- Higher occupancy helps hide memory latency
- Target: typically 30-50% minimum for good performance
- Limited by registers, shared memory, or block size
- Not always correlated with performance

#### **Memory Coalescing**
- Combining memory accesses from threads in a warp
- Consecutive threads accessing consecutive memory = optimal
- Misaligned access can reduce bandwidth by 32x
- 128-byte cache line granularity

#### **Arithmetic Intensity (AI)**
- Ratio of compute operations to memory operations (FLOPS/byte)
- High AI = compute-bound (good for GPUs)
- Low AI = memory-bound (bandwidth limited)
- Determines optimization strategy
- Roofline model visualizes performance limits

#### **Bank Conflicts**
- Multiple threads accessing same shared memory bank
- Can serialize access (up to 32x slower)
- 32 banks, 4-byte width each
- Formula: `bank = (address / 4) % 32`
- Avoided through padding or access pattern design

#### **IPC (Instructions Per Cycle)**
- Measures warp scheduler utilization
- Indicates how efficiently SM executes instructions
- Maximum varies by architecture (e.g., 4 for Volta)

## Key Relationships

### Compute vs Memory Bandwidth
- **Compute Capability**: H100 delivers 4000 TFLOPS (FP8 Tensor)
- **Memory Bandwidth**: 3 TB/sec HBM3
- **Balance**: Need ~1.3 bytes/FLOP for compute-bound operation
- Below this ratio → memory-bound, above → compute-bound

### Thread Block Scheduling
```
GPU → Multiple SMs → Each SM runs multiple blocks
Block → Multiple warps → Each warp has 32 threads
Warp → SIMT execution → All threads execute together
```

### Memory Access Hierarchy
```
Thread → Registers (private)
       → Shared Memory (block-level)
       → L1 Cache (SM-level)
       → L2 Cache (GPU-level)
       → Global Memory (device-level)
```

### Warp Execution Model
```
32 threads = 1 warp
All threads execute same instruction
Divergence → Serialization
Convergence → Parallel execution resumes
```

## Practical Examples

### Optimal Memory Access Pattern
```cuda
// Coalesced access - consecutive threads access consecutive memory
float value = data[blockIdx.x * blockDim.x + threadIdx.x];

// Strided access - poor performance
float value = data[threadIdx.x * stride];

// Random access - worst performance
float value = data[random_index[threadIdx.x]];
```

### Avoiding Bank Conflicts
```cuda
// Bank conflict - threads access same bank
__shared__ float tile[32][32];
float value = tile[threadIdx.x][0];  // All threads hit bank 0

// No conflict - with padding
__shared__ float tile[32][33];  // 33 = 32 + 1 padding
float value = tile[threadIdx.x][0];  // Different banks

// No conflict - stride access
float value = tile[0][threadIdx.x];  // Each thread different bank
```

### Occupancy Calculation
```cuda
// Block size affects occupancy
dim3 blockSize(256);  // 256 threads = 8 warps per block
// SM with 64 warp capacity → max 8 blocks (64/8)
// If using too many registers → fewer blocks can fit

// Example: 32 registers per thread
// SM has 65536 registers total
// Max threads = 65536/32 = 2048
// Max occupancy = 2048/2048 = 100%
```

### Kernel Launch Configuration
```cuda
// Calculate optimal launch configuration
int blockSize;  // Threads per block
int minGridSize; // Minimum number of blocks
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                   myKernel, 0, 0);

// Round up grid size
int gridSize = (N + blockSize - 1) / blockSize;
myKernel<<<gridSize, blockSize>>>(data, N);
```

## Additional Considerations

### Version-Specific Features
- **Compute Capability 9.0** (H100): Thread block clusters, TMA unit, 4th gen Tensor cores, FP8 support
- **Compute Capability 8.0** (A100): 3rd gen Tensor cores, 40 MB L2 cache, async copy
- **Compute Capability 7.0** (V100): 2nd gen Tensor cores, 6 MB L2 cache, independent thread scheduling
- **Compute Capability 6.0** (P100): Page migration engine, unified memory, NVLink

### Optimization Priorities
1. **Must Have**: 
   - Coalesced memory access
   - Avoid bank conflicts
   - Sufficient occupancy (>30%)
   - Minimize warp divergence

2. **Should Have**: 
   - Optimize arithmetic intensity
   - Use shared memory for data reuse
   - Overlap compute with memory transfers
   - Use appropriate precision (FP16/TF32 when possible)

3. **Nice to Have**: 
   - Fine-tune block sizes
   - Optimize register usage
   - Use texture memory for spatial data
   - Implement kernel fusion

### Common Pitfalls
- Assuming higher occupancy always means better performance
- Ignoring memory coalescing (can cause 32x slowdown)
- Not considering warp divergence in conditional code
- Overlooking shared memory bank conflicts
- Using wrong memory space for access pattern
- Not profiling before optimizing
- Over-optimizing compute-bound kernels when memory-bound

### Memory Bandwidth Calculation
```
Effective Bandwidth = (Bytes Read + Bytes Written) / Kernel Time
Bandwidth Efficiency = Effective Bandwidth / Theoretical Peak Bandwidth

Example for H100:
- Theoretical Peak: 3 TB/s
- Good utilization: >80% (2.4 TB/s)
- Memory-bound if bandwidth >90% utilized
```

## Performance Profiling Tools

### **nvidia-smi**
- Basic GPU utilization monitoring
- Power consumption and temperature
- Memory usage statistics
- Process information

### **Nsight Compute**
- Detailed kernel profiling
- Roofline analysis
- Memory access patterns
- Occupancy calculator
- Source-level analysis

### **Nsight Systems**
- System-wide performance analysis
- Timeline visualization
- CPU-GPU interaction
- Multi-GPU profiling

### **CUDA Occupancy Calculator**
- Spreadsheet or API-based
- Determines optimal launch configuration
- Shows limiting factors (registers, shared memory, blocks)

### **cudaMemcpy Profiling**
```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>(args);
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
```

## Glossary of Additional Terms

**SIMT**: Single Instruction, Multiple Thread - execution model where 32 threads execute same instruction

**GPC**: GPU Processing Cluster - group of SMs that share resources

**TPC**: Texture Processing Cluster - contains SMs and texture units

**TMA**: Tensor Memory Accelerator - hardware unit for efficient data movement (H100)

**MMA**: Matrix Multiply-Accumulate - core tensor operation

**GEMM**: General Matrix Multiply - fundamental operation in deep learning

**Roofline Model**: Performance model showing compute vs memory bandwidth limits

**Unified Memory**: Single memory space accessible by CPU and GPU

**Page Migration**: Automatic data movement between CPU and GPU memory

**NVLink**: High-speed GPU-to-GPU interconnect (up to 900 GB/s)