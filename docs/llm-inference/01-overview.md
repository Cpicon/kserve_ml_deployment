# LLM Inference Stack Overview: vLLM, llm-d, and KServe

## Introduction

This document provides a comprehensive overview of the modern Large Language Model (LLM) inference stack, focusing on three key technologies that work together to enable production-scale AI deployments. 

**Quick Summary**: vLLM provides the high-performance inference engine, llm-d orchestrates distributed serving on Kubernetes, and KServe offers the standardized deployment and management layer.

## Technology Stack Overview

```mermaid
graph TB
    subgraph "User Layer"
        Client[Client Applications]
        API[API Gateway]
    end
    
    subgraph "Management Layer - KServe"
        IS[InferenceService CRD]
        AS[Auto-scaling]
        Mon[Monitoring]
        AB[A/B Testing]
    end
    
    subgraph "Orchestration Layer - llm-d"
        IGW[Inference Gateway]
        Sched[vLLM-Aware Scheduler]
        Cache[Multi-Tier KV Cache]
    end
    
    subgraph "Inference Layer - vLLM"
        PA[PagedAttention]
        CB[Continuous Batching]
        TP[Tensor Parallelism]
        Model[LLM Model]
    end
    
    subgraph "Infrastructure"
        K8s[Kubernetes]
        GPU[GPUs/Accelerators]
    end
    
    Client --> API
    API --> IS
    IS --> IGW
    IGW --> Sched
    Sched --> PA
    PA --> CB
    CB --> TP
    TP --> Model
    Model --> GPU
    AS --> K8s
    Cache --> PA
    
    style Client fill:#e1f5fe
    style API fill:#e1f5fe
    style IS fill:#c8e6c9
    style IGW fill:#ffe0b2
    style PA fill:#ffccbc
    style GPU fill:#f5f5f5
```

## Key Components Explained

### 1. vLLM - The High-Performance Inference Engine

**vLLM** (Versatile Large Language Model) is the foundation of our stack, providing:

- **24x faster inference** than traditional transformers
- **Memory-efficient serving** through PagedAttention
- **Dynamic request handling** via continuous batching
- **Multi-GPU support** with tensor parallelism

[Learn more in the vLLM Deep Dive →](./02-vllm-deep-dive.md)

### 2. llm-d - Kubernetes-Native Distributed Inference

**llm-d** (LLM Daemon) orchestrates vLLM at scale:

- **Intelligent request routing** based on cache hits
- **Disaggregated serving** for prefill and decode stages
- **Multi-tier caching** for cost optimization
- **Native Kubernetes integration**

[Explore llm-d Architecture →](./03-llm-d-architecture.md)

### 3. KServe - Standardized Model Serving Platform

**KServe** provides the production-ready deployment layer:

- **Unified API** for all model frameworks
- **Auto-scaling** including scale-to-zero
- **Production features** like monitoring and A/B testing
- **Multi-framework support** beyond just LLMs

[Understand KServe Integration →](./04-kserve-integration.md)

## How They Work Together

```mermaid
sequenceDiagram
    participant User
    participant KServe
    participant llm-d
    participant vLLM
    participant GPU
    
    User->>KServe: Send inference request
    KServe->>KServe: Check auto-scaling rules
    KServe->>llm-d: Route to inference gateway
    llm-d->>llm-d: Check cache & load
    llm-d->>vLLM: Forward to best instance
    vLLM->>vLLM: PagedAttention processing
    vLLM->>GPU: Execute tensor operations
    GPU-->>vLLM: Return computations
    vLLM->>vLLM: Continuous batching
    vLLM-->>llm-d: Return tokens
    llm-d-->>KServe: Stream response
    KServe-->>User: Deliver inference result
```

## Performance Comparison

```mermaid
graph LR
    subgraph "Traditional Serving"
        T1[Request 1] --> TG1[GPU Idle Time]
        T2[Request 2] --> TG2[GPU Idle Time]
        T3[Request 3] --> TG3[GPU Idle Time]
    end
    
    subgraph "vLLM + llm-d + KServe"
        V1[Request 1] --> VG[Continuous GPU Utilization]
        V2[Request 2] --> VG
        V3[Request 3] --> VG
    end
    
    style TG1 fill:#ffcdd2
    style TG2 fill:#ffcdd2
    style TG3 fill:#ffcdd2
    style VG fill:#c8e6c9
```

## Use Cases and Benefits

### When to Use This Stack

✅ **Perfect for:**
- Production LLM deployments at scale
- Multi-tenant environments
- Cost-sensitive deployments
- High-throughput requirements
- Enterprise AI applications

### Key Benefits

| Benefit | Description | Impact |
|---------|-------------|---------|
| **Performance** | 24x throughput improvement | Handle more users with same hardware |
| **Cost Efficiency** | 90% better GPU utilization | Lower infrastructure costs |
| **Scalability** | Auto-scaling with KServe | Handle variable loads automatically |
| **Reliability** | Production-grade orchestration | 99.9%+ uptime achievable |
| **Flexibility** | Multi-model support | Deploy various model types |

## Quick Start Path

```mermaid
graph LR
    A[Choose Model] --> B[Setup Kubernetes]
    B --> C[Install KServe]
    C --> D[Deploy vLLM Runtime]
    D --> E[Configure llm-d]
    E --> F[Deploy InferenceService]
    F --> G[Test & Monitor]
    
    style A fill:#e3f2fd
    style G fill:#c8e6c9
```

## Next Steps

1. **[Deep Dive into vLLM](./02-vllm-deep-dive.md)** - Understand PagedAttention and performance optimizations
2. **[Explore llm-d Architecture](./03-llm-d-architecture.md)** - Learn about distributed inference patterns
3. **[Master KServe Integration](./04-kserve-integration.md)** - Deploy production-ready services
4. **[Study Architecture Patterns](./05-architecture-patterns.md)** - Choose the right deployment model
5. **[Follow Deployment Guide](./06-deployment-guide.md)** - Step-by-step production setup
6. **[Reference Glossary](./07-glossary.md)** - Technical terms and acronyms explained

## Industry Adoption

Major companies using this stack:
- **ByteDance** - AIBrix control plane
- **Amazon** - Rufus shopping assistant
- **LinkedIn** - AI features
- **LMSYS** - FastChat platform
- **Red Hat** - OpenShift AI

## Further Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [KServe Official Site](https://kserve.github.io/)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [Production Best Practices](./06-deployment-guide.md#best-practices)