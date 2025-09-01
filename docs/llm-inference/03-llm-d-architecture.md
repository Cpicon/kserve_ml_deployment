# llm-d Architecture: Kubernetes-Native Distributed LLM Inference

## What is llm-d?

**llm-d** (LLM Daemon) is a Kubernetes-native high-performance distributed Large Language Model inference framework launched by Red Hat in 2024. It addresses the critical need for scalable, efficient LLM inference in production environments by orchestrating vLLM instances intelligently across Kubernetes clusters.

## The Problem llm-d Solves

```mermaid
graph TB
    subgraph "Traditional LLM Deployment Challenges"
        C1[Inefficient Resource Usage]
        C2[No Cache Awareness]
        C3[Static Load Balancing]
        C4[Monolithic Serving]
        C5[Poor Multi-tenancy]
    end
    
    subgraph "Results"
        R1[High Costs]
        R2[Poor Performance]
        R3[Limited Scale]
    end
    
    C1 --> R1
    C2 --> R2
    C3 --> R2
    C4 --> R3
    C5 --> R3
    
    style R1 fill:#ffcdd2
    style R2 fill:#ffcdd2
    style R3 fill:#ffcdd2
```

## Core Architecture

```mermaid
graph TB
    subgraph "llm-d Architecture Stack"
        subgraph "Control Plane"
            CP[Controller Manager]
            Scheduler[vLLM-Aware Scheduler]
            CM[Cache Manager]
        end
        
        subgraph "Data Plane"
            IGW[Inference Gateway]
            Router[Smart Router]
            LB[Load Balancer]
        end
        
        subgraph "Inference Layer"
            subgraph "Prefill Stage"
                P1[Prefill Instance 1]
                P2[Prefill Instance 2]
            end
            
            subgraph "Decode Stage"
                D1[Decode Instance 1]
                D2[Decode Instance 2]
                D3[Decode Instance 3]
            end
        end
        
        subgraph "Storage Layer"
            KV1[L1 Cache - GPU]
            KV2[L2 Cache - CPU]
            KV3[L3 Cache - SSD]
        end
    end
    
    CP --> Scheduler
    Scheduler --> CM
    IGW --> Router
    Router --> LB
    LB --> P1
    LB --> P2
    P1 --> D1
    P2 --> D2
    P2 --> D3
    D1 --> KV1
    D2 --> KV1
    KV1 --> KV2
    KV2 --> KV3
```

## Key Components

### 1. Inference Gateway (IGW)

The Inference Gateway extends Kubernetes Gateway API with inference-specific capabilities:

```mermaid
sequenceDiagram
    participant Client
    participant IGW as Inference Gateway
    participant Scheduler
    participant Cache as Cache Index
    participant vLLM
    
    Client->>IGW: Inference Request
    IGW->>IGW: Parse request metadata
    IGW->>Cache: Check prefix cache
    Cache-->>IGW: Cache hit locations
    IGW->>Scheduler: Request routing decision
    Scheduler->>Scheduler: Evaluate load & cache
    Scheduler-->>IGW: Target instance
    IGW->>vLLM: Forward request
    vLLM-->>IGW: Stream tokens
    IGW-->>Client: Stream response
```

**Features:**
- **Request Classification**: Identifies prefill vs decode requests
- **Metadata Extraction**: Parses model, prompt prefix, parameters
- **Stream Management**: Handles token streaming efficiently
- **Protocol Translation**: Supports multiple inference protocols

### 2. vLLM-Aware Scheduler

```mermaid
graph LR
    subgraph "Scheduling Decision Factors"
        F1[Cache Hit Rate]
        F2[Current Load]
        F3[GPU Memory]
        F4[Queue Length]
        F5[Latency SLO]
    end
    
    subgraph "Scheduling Algorithm"
        A[Weight Calculation]
        B[Instance Ranking]
        C[Decision]
    end
    
    F1 --> A
    F2 --> A
    F3 --> A
    F4 --> A
    F5 --> A
    A --> B
    B --> C
    
    C --> I1[Instance Selection]
```

**Scheduling Strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Cache-Affinity** | Route to instances with warm cache | Repeated prompts |
| **Load-Balanced** | Distribute based on current load | General traffic |
| **Latency-Optimized** | Route to least loaded instance | Low-latency requirements |
| **Throughput-Optimized** | Maximize batch sizes | Bulk processing |

### 3. Multi-Tier KV Cache

```mermaid
graph TB
    subgraph "Cache Hierarchy"
        subgraph "Tier 1 - GPU HBM"
            T1[Active Sequences]
            Speed1[Access: 1-2ms]
            Size1[Size: 40-80GB]
        end
        
        subgraph "Tier 2 - CPU RAM"
            T2[Recent Sequences]
            Speed2[Access: 10-20ms]
            Size2[Size: 500GB-1TB]
        end
        
        subgraph "Tier 3 - NVMe SSD"
            T3[Historical Sequences]
            Speed3[Access: 50-100ms]
            Size3[Size: 10TB+]
        end
    end
    
    T1 -.Eviction.-> T2
    T2 -.Eviction.-> T3
    T3 -.Promotion.-> T2
    T2 -.Promotion.-> T1
    
    style T1 fill:#ffecb3
    style T2 fill:#e1f5fe
    style T3 fill:#f5f5f5
```

**Cache Management:**
- **LRU Eviction**: Least Recently Used policy
- **Prefix Deduplication**: Share common prefixes
- **Compression**: Quantize cached values
- **Preloading**: Warm cache for expected prompts

## Disaggregated Serving

### Traditional vs Disaggregated Architecture

```mermaid
graph LR
    subgraph "Traditional Monolithic"
        M[Single Instance] --> B[Both Prefill & Decode]
        B --> Problem[Interference & Inefficiency]
    end
    
    subgraph "llm-d Disaggregated"
        R[Request] --> P[Prefill Stage]
        P --> D[Decode Stage]
        P --> Optimize1[Optimized for Throughput]
        D --> Optimize2[Optimized for Latency]
    end
    
    style Problem fill:#ffcdd2
    style Optimize1 fill:#c8e6c9
    style Optimize2 fill:#c8e6c9
```

### Prefill Stage Optimization

```mermaid
graph TB
    subgraph "Prefill Characteristics"
        PC1[Compute Intensive]
        PC2[Large Batch Friendly]
        PC3[No State Dependency]
        PC4[Parallelizable]
    end
    
    subgraph "Optimization"
        O1[Large Batch Sizes]
        O2[Higher GPU Utilization]
        O3[Tensor Parallelism]
    end
    
    PC1 --> O1
    PC2 --> O1
    PC3 --> O2
    PC4 --> O3
```

### Decode Stage Optimization

```mermaid
graph TB
    subgraph "Decode Characteristics"
        DC1[Memory Bound]
        DC2[Sequential]
        DC3[State Dependent]
        DC4[Latency Sensitive]
    end
    
    subgraph "Optimization"
        O1[Continuous Batching]
        O2[KV Cache Optimization]
        O3[Speculative Decoding]
    end
    
    DC1 --> O2
    DC2 --> O1
    DC3 --> O2
    DC4 --> O3
```

## Request Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant Gateway as Inference Gateway
    participant Scheduler
    participant Prefill as Prefill Instance
    participant Decode as Decode Instance
    participant Cache as KV Cache
    
    User->>Gateway: Submit prompt
    Gateway->>Scheduler: Route request
    Scheduler->>Scheduler: Check cache & load
    Scheduler->>Prefill: Assign to prefill
    Prefill->>Cache: Generate KV cache
    Cache-->>Prefill: Cache stored
    Prefill->>Decode: Transfer to decode
    
    loop Token Generation
        Decode->>Cache: Read KV cache
        Cache-->>Decode: Cache data
        Decode->>Decode: Generate token
        Decode-->>Gateway: Stream token
        Gateway-->>User: Send token
    end
```

## Deployment Patterns

### 1. Basic Deployment

```yaml
apiVersion: llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-deployment
spec:
  model:
    name: meta-llama/Llama-2-7b-hf
    runtime: vllm
  serving:
    mode: unified  # Both prefill and decode
    replicas: 3
  resources:
    gpu: nvidia.com/gpu
    count: 1
    memory: 40Gi
```

### 2. Disaggregated Deployment

```yaml
apiVersion: llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-disaggregated
spec:
  model:
    name: meta-llama/Llama-2-70b-hf
    runtime: vllm
  serving:
    mode: disaggregated
    prefill:
      replicas: 2
      resources:
        gpu: 2
        memory: 80Gi
    decode:
      replicas: 6
      resources:
        gpu: 1
        memory: 40Gi
  cache:
    tiers:
      - type: gpu
        size: 40Gi
      - type: cpu
        size: 500Gi
      - type: ssd
        size: 2Ti
```

## Performance Benefits

### Throughput Improvements

```mermaid
graph LR
    subgraph "Performance Gains"
        B[Baseline] --> I1[Cache-Aware: +40%]
        I1 --> I2[Disaggregated: +60%]
        I2 --> I3[Multi-Tier Cache: +30%]
        I3 --> Total[Total: 3.5x Improvement]
    end
    
    style Total fill:#4caf50,color:#fff
```

### Resource Efficiency

| Metric | Traditional | llm-d | Improvement |
|--------|------------|-------|-------------|
| GPU Utilization | 40-50% | 85-95% | 2x |
| Memory Efficiency | 60% | 95% | 1.6x |
| Cost per Token | $0.01 | $0.003 | 3.3x |
| Latency P99 | 2000ms | 500ms | 4x |

## Integration with Kubernetes

### Custom Resources

```mermaid
graph TB
    subgraph "llm-d CRDs"
        LLMDeployment[LLMDeployment]
        InferenceGateway[InferenceGateway]
        CachePolicy[CachePolicy]
        SchedulingPolicy[SchedulingPolicy]
    end
    
    subgraph "Native K8s Resources"
        Deployment[Deployment]
        Service[Service]
        ConfigMap[ConfigMap]
        PVC[PersistentVolumeClaim]
    end
    
    LLMDeployment --> Deployment
    InferenceGateway --> Service
    CachePolicy --> ConfigMap
    SchedulingPolicy --> ConfigMap
```

### Operator Pattern

```mermaid
sequenceDiagram
    participant User
    participant K8s as Kubernetes API
    participant Op as llm-d Operator
    participant Ctrl as Controller
    participant Resources
    
    User->>K8s: Create LLMDeployment
    K8s->>Op: Watch event
    Op->>Ctrl: Process resource
    Ctrl->>Ctrl: Generate manifests
    Ctrl->>K8s: Create resources
    K8s->>Resources: Deploy pods/services
    Resources-->>Op: Status update
    Op-->>K8s: Update status
    K8s-->>User: Deployment ready
```

## Monitoring and Observability

### Key Metrics

```mermaid
graph LR
    subgraph "llm-d Metrics"
        subgraph "Performance"
            M1[Tokens/sec]
            M2[Time to First Token]
            M3[Request Latency]
        end
        
        subgraph "Resource"
            M4[GPU Utilization]
            M5[Cache Hit Rate]
            M6[Memory Usage]
        end
        
        subgraph "Reliability"
            M7[Error Rate]
            M8[Queue Length]
            M9[Timeout Rate]
        end
    end
    
    M1 --> Prometheus
    M2 --> Prometheus
    M3 --> Prometheus
    M4 --> Prometheus
    M5 --> Prometheus
    M6 --> Prometheus
    M7 --> Prometheus
    M8 --> Prometheus
    M9 --> Prometheus
    
    Prometheus --> Grafana[Grafana Dashboard]
```

## Production Deployment Considerations

### High Availability Setup

```mermaid
graph TB
    subgraph "HA Architecture"
        subgraph "Zone A"
            GW1[Gateway]
            P1[Prefill]
            D1[Decode]
        end
        
        subgraph "Zone B"
            GW2[Gateway]
            P2[Prefill]
            D2[Decode]
        end
        
        subgraph "Zone C"
            GW3[Gateway]
            P3[Prefill]
            D3[Decode]
        end
        
        LB[Load Balancer]
        Cache[Distributed Cache]
    end
    
    LB --> GW1
    LB --> GW2
    LB --> GW3
    
    GW1 --> P1
    GW2 --> P2
    GW3 --> P3
    
    P1 --> Cache
    P2 --> Cache
    P3 --> Cache
```

### Scaling Strategies

| Strategy | Trigger | Action | Use Case |
|----------|---------|--------|----------|
| **Horizontal Pod Autoscaling** | CPU/Memory | Add pods | General load |
| **Vertical Pod Autoscaling** | Resource limits | Resize pods | Right-sizing |
| **Cluster Autoscaling** | Node capacity | Add nodes | Peak traffic |
| **Predictive Scaling** | Traffic patterns | Pre-scale | Known patterns |

## Industry Support and Ecosystem

### Founding Contributors
- **Red Hat**: Project initiator and lead
- **CoreWeave**: Cloud infrastructure
- **Google Cloud**: Kubernetes expertise
- **IBM Research**: AI research
- **NVIDIA**: GPU optimization

### Technology Partners
- **AMD**: GPU support
- **Cisco**: Networking
- **Hugging Face**: Model hub
- **Intel**: CPU optimization
- **Lambda**: Cloud compute
- **Mistral AI**: Model provider

## Future Roadmap

### 2025 Planned Features
- **Enhanced multi-cloud support**
- **Federated inference**
- **Advanced caching algorithms**
- **Native function calling**
- **Improved observability**
- **Cost optimization features**

## Summary

llm-d transforms LLM serving through:
- **Intelligent routing** with cache awareness
- **Disaggregated serving** for efficiency
- **Multi-tier caching** for cost optimization
- **Native Kubernetes integration**
- **Production-grade reliability**

Next: [KServe Integration →](./04-kserve-integration.md)