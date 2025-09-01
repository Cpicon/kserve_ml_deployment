# Architecture Patterns for LLM Inference

## Overview

This document explores various architectural patterns for deploying Large Language Models using vLLM, llm-d, and KServe, helping you choose the right approach for your use case.

## Pattern Decision Matrix

```mermaid
graph TB
    Start[Start] --> Size{Model Size?}
    
    Size -->|< 7B| Small[Small Models]
    Size -->|7B - 30B| Medium[Medium Models]
    Size -->|> 30B| Large[Large Models]
    
    Small --> Traffic1{Traffic Pattern?}
    Medium --> Traffic2{Traffic Pattern?}
    Large --> Traffic3{Traffic Pattern?}
    
    Traffic1 -->|Sporadic| P1[Single Instance]
    Traffic1 -->|Steady| P2[Replicated Instances]
    
    Traffic2 -->|Variable| P3[Auto-scaling Pool]
    Traffic2 -->|High| P4[Disaggregated Serving]
    
    Traffic3 -->|Any| P5[Tensor Parallel + Disaggregated]
    
    style P1 fill:#e8f5e9
    style P2 fill:#e8f5e9
    style P3 fill:#fff3e0
    style P4 fill:#fff3e0
    style P5 fill:#ffebee
```

## Pattern 1: Single Instance Serving

### Architecture

```mermaid
graph LR
    subgraph "Single Instance Pattern"
        Client[Clients] --> LB[Load Balancer]
        LB --> Instance[vLLM Instance]
        Instance --> GPU[Single GPU]
        Instance --> Storage[Model Storage]
    end
    
    style Instance fill:#c8e6c9
```

### Use Cases
- **Development and testing environments**
- **Small models (< 7B parameters)**
- **Low traffic applications**
- **Cost-sensitive deployments**

### Configuration Example

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: single-instance-llm
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllmserver
      storageUri: s3://models/llama-2-7b
    minReplicas: 1
    maxReplicas: 1  # Fixed single instance
    resources:
      requests:
        nvidia.com/gpu: "1"
        memory: "30Gi"
```

### Pros and Cons

| Pros | Cons |
|------|------|
| ✅ Simple to deploy | ❌ No redundancy |
| ✅ Low cost | ❌ Limited throughput |
| ✅ Easy to debug | ❌ Single point of failure |
| ✅ Minimal overhead | ❌ No scale capability |

## Pattern 2: Replicated Instances

### Architecture

```mermaid
graph TB
    subgraph "Replicated Pattern"
        Client[Clients] --> LB[Load Balancer]
        LB --> I1[vLLM Instance 1]
        LB --> I2[vLLM Instance 2]
        LB --> I3[vLLM Instance 3]
        
        I1 --> GPU1[GPU 1]
        I2 --> GPU2[GPU 2]
        I3 --> GPU3[GPU 3]
        
        Storage[Shared Model Storage]
        I1 --> Storage
        I2 --> Storage
        I3 --> Storage
    end
    
    style I1 fill:#c8e6c9
    style I2 fill:#c8e6c9
    style I3 fill:#c8e6c9
```

### Implementation

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: replicated-llm
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllmserver
      storageUri: s3://models/mistral-7b
    minReplicas: 3
    maxReplicas: 3  # Fixed replication
    resources:
      requests:
        nvidia.com/gpu: "1"
        memory: "40Gi"
```

### Load Balancing Strategies

```mermaid
graph LR
    subgraph "Load Balancing Options"
        RR[Round Robin] --> Desc1[Equal distribution]
        LC[Least Connections] --> Desc2[Route to least busy]
        WRR[Weighted Round Robin] --> Desc3[Performance-based]
        CH[Consistent Hashing] --> Desc4[Session affinity]
    end
```

## Pattern 3: Auto-scaling Pool

### Architecture

```mermaid
graph TB
    subgraph "Auto-scaling Pattern"
        Client[Clients] --> Gateway[API Gateway]
        Gateway --> KServe[KServe Controller]
        
        subgraph "Dynamic Pool"
            Min[Min: 2 Pods]
            Current[Current: 5 Pods]
            Max[Max: 20 Pods]
        end
        
        KServe --> Metrics[Metrics Server]
        Metrics --> HPA[HPA/KPA]
        HPA --> Current
        
        Current --> Scale{Scale Decision}
        Scale -->|High Load| ScaleUp[Add Pods]
        Scale -->|Low Load| ScaleDown[Remove Pods]
    end
    
    style Current fill:#fff3e0
```

### Configuration

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: autoscaling-llm
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllmserver
      storageUri: s3://models/llama-2-13b
    minReplicas: 2
    maxReplicas: 20
    scaleTarget: 60  # Target utilization %
    scaleMetric: concurrency
    containerConcurrency: 10
    resources:
      requests:
        nvidia.com/gpu: "1"
        memory: "50Gi"
```

### Scaling Behavior

```mermaid
graph LR
    subgraph "Scaling Timeline"
        T0[00:00<br/>2 pods] --> T1[00:05<br/>Traffic spike]
        T1 --> T2[00:06<br/>Scale to 5]
        T2 --> T3[00:10<br/>Scale to 10]
        T3 --> T4[00:20<br/>Traffic drops]
        T4 --> T5[00:25<br/>Scale to 5]
        T5 --> T6[00:35<br/>Scale to 2]
    end
    
    style T1 fill:#ffcdd2
    style T4 fill:#c8e6c9
```

## Pattern 4: Disaggregated Serving

### Architecture

```mermaid
graph TB
    subgraph "Disaggregated Architecture"
        Client[Clients] --> Router[Smart Router]
        
        Router --> PrefillLB[Prefill LB]
        Router --> DecodeLB[Decode LB]
        
        subgraph "Prefill Stage"
            P1[Prefill 1<br/>Batch Optimized]
            P2[Prefill 2<br/>Batch Optimized]
        end
        
        subgraph "Decode Stage"
            D1[Decode 1<br/>Latency Optimized]
            D2[Decode 2<br/>Latency Optimized]
            D3[Decode 3<br/>Latency Optimized]
        end
        
        PrefillLB --> P1
        PrefillLB --> P2
        
        DecodeLB --> D1
        DecodeLB --> D2
        DecodeLB --> D3
        
        P1 --> Cache[Shared KV Cache]
        P2 --> Cache
        Cache --> D1
        Cache --> D2
        Cache --> D3
    end
    
    style P1 fill:#e1f5fe
    style P2 fill:#e1f5fe
    style D1 fill:#fff3e0
    style D2 fill:#fff3e0
    style D3 fill:#fff3e0
```

### Implementation with llm-d

```yaml
apiVersion: llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: disaggregated-llm
spec:
  model:
    name: meta-llama/Llama-2-30b-hf
    runtime: vllm
  serving:
    mode: disaggregated
    prefill:
      replicas: 2
      batchSize: 32
      resources:
        gpu: 2
        memory: 80Gi
    decode:
      replicas: 5
      batchSize: 8
      resources:
        gpu: 1
        memory: 40Gi
  cache:
    type: distributed
    size: 500Gi
```

### Performance Characteristics

| Stage | Optimization | Benefit |
|-------|-------------|---------|
| **Prefill** | Large batches | 3x throughput |
| **Decode** | Low latency | 2x faster TTFT |
| **Cache** | Shared across stages | 40% memory savings |

## Pattern 5: Tensor Parallel Serving

### Architecture

```mermaid
graph TB
    subgraph "Tensor Parallel Pattern"
        Client[Clients] --> Gateway[Gateway]
        
        subgraph "Model Shard 1"
            GPU1[GPU 1<br/>Layers 0-10]
            GPU2[GPU 2<br/>Layers 11-20]
            GPU3[GPU 3<br/>Layers 21-30]
            GPU4[GPU 4<br/>Layers 31-40]
        end
        
        Gateway --> GPU1
        GPU1 <--> GPU2
        GPU2 <--> GPU3
        GPU3 <--> GPU4
        
        GPU4 --> Output[Output]
        
        AllReduce[AllReduce<br/>Communication]
        GPU1 --> AllReduce
        GPU2 --> AllReduce
        GPU3 --> AllReduce
        GPU4 --> AllReduce
    end
    
    style GPU1 fill:#ffecb3
    style GPU2 fill:#ffecb3
    style GPU3 fill:#ffecb3
    style GPU4 fill:#ffecb3
```

### Configuration

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: tensor-parallel-llm
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      runtime: kserve-vllmserver
      storageUri: s3://models/llama-2-70b
      env:
      - name: TENSOR_PARALLEL_SIZE
        value: "4"
      - name: PIPELINE_PARALLEL_SIZE
        value: "1"
    minReplicas: 1
    maxReplicas: 5
    resources:
      requests:
        nvidia.com/gpu: "4"  # 4 GPUs per replica
        memory: "320Gi"
```

## Pattern 6: Hybrid Multi-Tier

### Architecture

```mermaid
graph TB
    subgraph "Hybrid Multi-Tier Pattern"
        Client[Clients] --> Classifier[Request Classifier]
        
        Classifier --> Simple{Complexity?}
        
        Simple -->|Simple| SmallModel[7B Model<br/>Fast Response]
        Simple -->|Medium| MediumModel[13B Model<br/>Balanced]
        Simple -->|Complex| LargeModel[70B Model<br/>High Quality]
        
        subgraph "Tier 1 - Edge"
            SmallModel --> Edge[Edge GPUs]
        end
        
        subgraph "Tier 2 - Regional"
            MediumModel --> Regional[Regional Clusters]
        end
        
        subgraph "Tier 3 - Central"
            LargeModel --> Central[Central Data Center]
        end
    end
    
    style SmallModel fill:#e8f5e9
    style MediumModel fill:#fff3e0
    style LargeModel fill:#ffebee
```

### Routing Logic

```python
# Pseudo-code for request routing
def route_request(prompt, requirements):
    complexity = analyze_complexity(prompt)
    latency_requirement = requirements.get('max_latency_ms')
    quality_requirement = requirements.get('min_quality_score')
    
    if latency_requirement < 100:
        return "edge-7b-model"
    elif complexity == "simple" and quality_requirement < 0.8:
        return "regional-13b-model"
    else:
        return "central-70b-model"
```

## Pattern 7: Federation Pattern

### Architecture

```mermaid
graph TB
    subgraph "Federated Deployment"
        subgraph "Region US-East"
            USE[US-East Cluster]
            USEM[Models A, B]
        end
        
        subgraph "Region EU-West"
            EUW[EU-West Cluster]
            EUWM[Models B, C]
        end
        
        subgraph "Region APAC"
            APAC[APAC Cluster]
            APACM[Models A, C]
        end
        
        Global[Global Router]
        
        Global --> USE
        Global --> EUW
        Global --> APAC
        
        USE --> USEM
        EUW --> EUWM
        APAC --> APACM
        
        Registry[Global Model Registry]
        USE --> Registry
        EUW --> Registry
        APAC --> Registry
    end
```

### Benefits

- **Data sovereignty compliance**
- **Reduced latency for regional users**
- **Disaster recovery capability**
- **Cost optimization via regional pricing**

## Pattern 8: Caching-First Architecture

### Architecture

```mermaid
graph TB
    subgraph "Caching-First Pattern"
        Request[Request] --> CacheCheck{Cache Hit?}
        
        CacheCheck -->|Yes| L1[L1 Cache<br/>GPU Memory]
        CacheCheck -->|No| CheckL2{Check L2}
        
        L1 --> FastResponse[< 10ms Response]
        
        CheckL2 -->|Hit| L2[L2 Cache<br/>CPU Memory]
        CheckL2 -->|Miss| CheckL3{Check L3}
        
        L2 --> MediumResponse[< 50ms Response]
        
        CheckL3 -->|Hit| L3[L3 Cache<br/>SSD Storage]
        CheckL3 -->|Miss| Compute[Full Computation]
        
        L3 --> SlowResponse[< 200ms Response]
        Compute --> NormalResponse[Normal Latency]
        
        Compute --> UpdateCache[Update All Caches]
    end
    
    style L1 fill:#c8e6c9
    style L2 fill:#fff3e0
    style L3 fill:#ffecb3
```

### Cache Strategy

```yaml
apiVersion: llm-d.io/v1alpha1
kind: CachePolicy
metadata:
  name: multi-tier-cache
spec:
  tiers:
  - name: l1-gpu
    type: gpu
    size: 40Gi
    eviction: lru
    ttl: 1h
  - name: l2-cpu
    type: memory
    size: 500Gi
    eviction: lfu
    ttl: 6h
  - name: l3-ssd
    type: disk
    size: 5Ti
    eviction: fifo
    ttl: 24h
  prefixSharing:
    enabled: true
    minLength: 10
```

## Pattern Selection Guide

### Decision Criteria

```mermaid
graph LR
    subgraph "Selection Factors"
        F1[Model Size] --> Pattern
        F2[Traffic Volume] --> Pattern
        F3[Latency Requirements] --> Pattern
        F4[Cost Budget] --> Pattern
        F5[Availability Needs] --> Pattern
    end
    
    Pattern[Selected Pattern]
```

### Pattern Comparison Matrix

| Pattern | Model Size | Traffic | Latency | Cost | Complexity |
|---------|------------|---------|---------|------|------------|
| **Single Instance** | < 7B | Low | Medium | $ | Low |
| **Replicated** | < 13B | Medium | Low | $$ | Low |
| **Auto-scaling** | < 30B | Variable | Low | $$$ | Medium |
| **Disaggregated** | Any | High | Very Low | $$$ | High |
| **Tensor Parallel** | > 30B | Any | Medium | $$$$ | High |
| **Hybrid Multi-Tier** | Mixed | High | Variable | $$$$ | Very High |
| **Federation** | Any | Global | Low | $$$$$ | Very High |
| **Caching-First** | Any | Repeated | Very Low | $$ | Medium |

## Implementation Best Practices

### 1. Start Simple, Scale Gradually

```mermaid
graph LR
    Stage1[Single Instance] --> Stage2[Replicated]
    Stage2 --> Stage3[Auto-scaling]
    Stage3 --> Stage4[Disaggregated]
    Stage4 --> Stage5[Hybrid]
    
    style Stage1 fill:#e8f5e9
    style Stage5 fill:#ffebee
```

### 2. Monitor Key Metrics

- **Latency**: P50, P95, P99
- **Throughput**: Tokens/second
- **Utilization**: GPU, Memory, Network
- **Errors**: Rate, types, recovery
- **Cost**: Per token, per request

### 3. Optimize for Your Use Case

| Use Case | Recommended Pattern | Key Optimization |
|----------|-------------------|------------------|
| **Chatbot** | Auto-scaling | Low latency |
| **Batch Processing** | Replicated | High throughput |
| **Real-time API** | Disaggregated | Consistent latency |
| **Research** | Single Instance | Cost efficiency |
| **Enterprise** | Hybrid Multi-Tier | Flexibility |

## Anti-Patterns to Avoid

### 1. Over-Engineering

❌ **Don't**: Start with complex patterns for simple use cases
✅ **Do**: Begin with simple patterns and evolve based on needs

### 2. Ignoring Cache Potential

❌ **Don't**: Compute everything from scratch
✅ **Do**: Implement intelligent caching strategies

### 3. Fixed Scaling

❌ **Don't**: Use fixed replicas for variable traffic
✅ **Do**: Implement auto-scaling based on actual metrics

### 4. Single Point of Failure

❌ **Don't**: Deploy single instances for production
✅ **Do**: Ensure redundancy and failover capabilities

## Summary

Choose your architecture pattern based on:
- **Model size and complexity**
- **Traffic patterns and volume**
- **Latency and throughput requirements**
- **Budget constraints**
- **Operational complexity tolerance**

Start simple, measure everything, and evolve your architecture as needs grow.

Next: [Deployment Guide →](./06-deployment-guide.md)