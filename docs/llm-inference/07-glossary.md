# Glossary and Technical Terminology

## Core Technologies

### vLLM
**Versatile Large Language Model** - A high-performance inference and serving library for Large Language Models, providing up to 24x throughput improvement over traditional transformers through innovations like PagedAttention and continuous batching.

### llm-d
**LLM Daemon** - A Kubernetes-native distributed inference framework launched by Red Hat in 2024, designed to orchestrate vLLM instances intelligently across clusters with features like cache-aware routing and disaggregated serving.

### KServe
**Kubernetes Serving** - A standard, cloud-native model serving platform on Kubernetes that provides a Custom Resource Definition (CRD) for deploying and managing machine learning models at scale.

## Architecture Components

### A

**AllReduce**
A collective communication operation used in distributed computing where all processes contribute data that gets reduced (combined) and the result is distributed back to all processes. Critical for tensor parallelism synchronization.

**Auto-scaling**
The automatic adjustment of computational resources (pods, replicas) based on metrics like CPU usage, memory, or custom metrics such as request concurrency.

**AWQ (Activation-aware Weight Quantization)**
A quantization method that preserves important weights based on activation patterns, enabling 4-bit quantization with minimal accuracy loss.

### B

**Batch Size**
The number of input samples processed together in a single forward pass through the model. Larger batch sizes improve throughput but require more memory.

**Block Manager**
Component in vLLM responsible for allocating and managing memory blocks for the KV cache, implementing the PagedAttention algorithm.

### C

**Canary Deployment**
A deployment pattern where a new version is rolled out to a small subset of users before full deployment, enabling testing in production with minimal risk.

**Continuous Batching**
A dynamic batching technique in vLLM where requests are continuously grouped and processed without waiting for fixed batch sizes, maximizing GPU utilization.

**CRD (Custom Resource Definition)**
A Kubernetes extension mechanism that allows users to create custom resources. KServe uses CRDs like InferenceService to define model deployments.

**CUDA**
**Compute Unified Device Architecture** - NVIDIA's parallel computing platform and API model for GPU computing.

### D

**Decode Stage**
The phase in LLM inference where tokens are generated one by one based on the initial context. This stage is memory-bound and benefits from KV cache optimization.

**Disaggregated Serving**
An architecture pattern where prefill and decode stages of LLM inference are handled by separate, specialized instances for optimal resource utilization.

**DP (Data Parallelism)**
A parallelization strategy where the same model is replicated across multiple devices, each processing different batches of data.

### E

**Eviction Policy**
Rules determining which cached items to remove when cache capacity is reached. Common policies include LRU (Least Recently Used), LFU (Least Frequently Used), and FIFO (First In, First Out).

**Explainer**
A KServe component that provides model interpretability by explaining predictions, useful for understanding why a model made specific decisions.

### F

**FP16/FP8 (Floating Point 16/8-bit)**
Reduced precision number formats that use 16 or 8 bits instead of 32 bits, reducing memory usage and improving computation speed with minimal accuracy impact.

**FlashAttention**
An optimized attention mechanism that reduces memory usage and improves speed by computing attention in a block-wise manner with minimal memory access.

### G

**Gateway API**
A Kubernetes API for managing ingress traffic, extended by llm-d's Inference Gateway to provide inference-specific routing capabilities.

**GEMM (General Matrix Multiply)**
Fundamental operation in deep learning for matrix multiplication, heavily optimized in GPU libraries like cuBLAS.

**GPTQ (Generative Pre-trained Transformer Quantization)**
A post-training quantization method specifically designed for transformer models, enabling 4-bit quantization.

**GPU (Graphics Processing Unit)**
Specialized processor originally designed for graphics but now essential for parallel computing tasks like deep learning inference.

**GPU Memory Utilization**
Percentage of GPU memory allocated for model and cache storage. vLLM typically sets this to 0.90-0.95 for maximum efficiency.

### H

**HBM (High Bandwidth Memory)**
High-performance memory used in modern GPUs, providing much higher bandwidth than traditional DRAM.

**HPA (Horizontal Pod Autoscaler)**
Kubernetes component that automatically scales the number of pods based on observed CPU/memory utilization or custom metrics.

### I

**InferenceService**
The primary Custom Resource in KServe for deploying machine learning models, defining predictor, transformer, and explainer components.

**Inference Gateway (IGW)**
llm-d component that extends Kubernetes Gateway API with inference-specific routing capabilities like cache-aware scheduling.

**Istio**
A service mesh that provides traffic management, security, and observability for microservices, often used with KServe for advanced networking features.

### J

**JWT (JSON Web Token)**
A compact, URL-safe means of representing claims to be transferred between parties, commonly used for API authentication.

### K

**Knative**
A Kubernetes-based platform for deploying and managing serverless workloads, providing auto-scaling and scale-to-zero capabilities for KServe.

**KPA (Knative Pod Autoscaler)**
Knative's autoscaler that can scale based on request concurrency and supports scale-to-zero, unlike standard HPA.

**KV Cache (Key-Value Cache)**
Storage for attention computation results (keys and values) that are reused across token generation steps, critical for LLM inference efficiency.

**Kubernetes (K8s)**
An open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.

### L

**Latency**
The time delay between sending a request and receiving a response. In LLM context, often measured as Time to First Token (TTFT) and Inter-Token Latency (ITL).

**LLM (Large Language Model)**
Neural network models with billions of parameters trained on vast text datasets, capable of understanding and generating human-like text.

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning method that adds small trainable matrices to frozen model weights, enabling multiple adaptations to be served simultaneously.

**LRU (Least Recently Used)**
Cache eviction policy that removes the least recently accessed items first when the cache reaches capacity.

### M

**Max Model Length**
Maximum number of tokens (input + output) that a model can process in a single request, limited by available memory and model architecture.

**Megatron-LM**
A framework for training large transformer models using model parallelism techniques, whose algorithms are used in vLLM's tensor parallelism.

**Model Mesh**
A framework for serving multiple models on a shared pool of resources, optimizing resource utilization across different models.

**mTLS (Mutual TLS)**
A security protocol where both client and server authenticate each other using certificates, providing end-to-end encryption.

### N

**Network Policy**
Kubernetes resource that controls network traffic between pods, namespaces, and external endpoints for security isolation.

**NVLink**
NVIDIA's high-speed interconnect for direct GPU-to-GPU communication, crucial for multi-GPU tensor parallelism.

### O

**ONNX (Open Neural Network Exchange)**
An open format for representing machine learning models, enabling interoperability between different frameworks.

**OpenAI API**
A REST API specification for interacting with language models, widely adopted as a standard interface that vLLM supports.

### P

**PagedAttention**
vLLM's core algorithm that manages KV cache like virtual memory in operating systems, storing attention keys and values in non-contiguous memory blocks.

**Pipeline Parallelism (PP)**
A parallelization strategy where different layers of a model are placed on different devices, with data flowing through them in a pipeline fashion.

**Predictor**
The primary component in a KServe InferenceService responsible for running model inference.

**Prefill Stage**
The initial phase of LLM inference where the entire input prompt is processed to generate the KV cache. This stage is compute-intensive.

**Prometheus**
An open-source monitoring system that collects metrics from configured targets at intervals, evaluates rule expressions, and can trigger alerts.

### Q

**QPS (Queries Per Second)**
Metric measuring the rate at which a system processes requests, important for capacity planning and performance monitoring.

**Quantization**
The process of reducing the precision of model weights and activations from floating-point to lower-bit representations to reduce memory and improve speed.

### R

**RBAC (Role-Based Access Control)**
Kubernetes security feature that regulates access to resources based on the roles of individual users or service accounts.

**Replica**
A copy of a pod or deployment in Kubernetes, used for load balancing and high availability.

**RPS (Requests Per Second)**
Similar to QPS, measuring the incoming request rate to a service.

### S

**Sampling Parameters**
Configuration for text generation including temperature (randomness), top_p (nucleus sampling), top_k (top-k sampling), and max_tokens.

**Scale-to-Zero**
Capability to automatically scale down to zero replicas when there's no traffic, saving resources. Supported by Knative but not standard Kubernetes HPA.

**Service Mesh**
Infrastructure layer for managing service-to-service communication, providing features like load balancing, authentication, and monitoring. Istio is a popular example.

**SLO (Service Level Objective)**
A target value or range for service performance metrics like latency, throughput, or availability.

**Speculative Decoding**
Technique where a smaller "draft" model generates multiple tokens quickly, which are then verified by the main model in parallel, reducing overall latency.

**StorageUri**
In KServe, the location where model artifacts are stored, supporting various backends like S3, GCS, Azure Blob, or HTTP endpoints.

### T

**Tensor Parallelism (TP)**
Model parallelism technique where individual operations (like matrix multiplications) are split across multiple devices.

**Throughput**
The amount of data or number of tokens processed per unit time, typically measured in tokens per second for LLMs.

**Token**
Basic unit of text in LLMs, typically representing a word, subword, or character. Models have fixed vocabularies mapping tokens to integers.

**Transformer**
(1) Neural network architecture using self-attention mechanisms, foundation for modern LLMs
(2) KServe component for pre/post-processing of model inputs/outputs

**TTFT (Time To First Token)**
Latency metric measuring the time from request submission to generation of the first output token, critical for user experience.

### V

**Virtual Service**
Istio resource that defines routing rules for how requests are handled within the service mesh.

**VRAM (Video RAM)**
Memory on graphics cards used for storing model weights, activations, and KV cache during inference.

### W

**Webhook**
HTTP callback that triggers actions in response to events. Used in Kubernetes for admission control and validation.

**Weight Quantization**
Process of reducing the precision of model weights to decrease memory usage and potentially improve inference speed.

### Acronyms Quick Reference

| Acronym | Full Form | Category |
|---------|-----------|----------|
| **API** | Application Programming Interface | General |
| **AWS** | Amazon Web Services | Cloud |
| **CLI** | Command Line Interface | Tools |
| **CPU** | Central Processing Unit | Hardware |
| **CRD** | Custom Resource Definition | Kubernetes |
| **CUDA** | Compute Unified Device Architecture | GPU |
| **DNS** | Domain Name System | Networking |
| **GCS** | Google Cloud Storage | Cloud |
| **GPU** | Graphics Processing Unit | Hardware |
| **HA** | High Availability | Architecture |
| **HPA** | Horizontal Pod Autoscaler | Kubernetes |
| **HTTP** | Hypertext Transfer Protocol | Networking |
| **HTTPS** | HTTP Secure | Networking |
| **JSON** | JavaScript Object Notation | Data Format |
| **JWT** | JSON Web Token | Security |
| **K8s** | Kubernetes | Platform |
| **KPA** | Knative Pod Autoscaler | Kubernetes |
| **KV** | Key-Value | Data Structure |
| **LLM** | Large Language Model | AI/ML |
| **ML** | Machine Learning | AI/ML |
| **mTLS** | Mutual TLS | Security |
| **NLP** | Natural Language Processing | AI/ML |
| **ONNX** | Open Neural Network Exchange | ML Format |
| **OOM** | Out of Memory | Error |
| **P2P** | Peer to Peer | Networking |
| **PVC** | PersistentVolumeClaim | Kubernetes |
| **QPS** | Queries Per Second | Metrics |
| **RAM** | Random Access Memory | Hardware |
| **RBAC** | Role-Based Access Control | Security |
| **REST** | Representational State Transfer | API |
| **RPC** | Remote Procedure Call | Protocol |
| **RPS** | Requests Per Second | Metrics |
| **S3** | Simple Storage Service | Cloud |
| **SDK** | Software Development Kit | Tools |
| **SLA** | Service Level Agreement | Operations |
| **SLO** | Service Level Objective | Operations |
| **SSD** | Solid State Drive | Hardware |
| **SSL** | Secure Sockets Layer | Security |
| **TCP** | Transmission Control Protocol | Networking |
| **TLS** | Transport Layer Security | Security |
| **TTFT** | Time To First Token | Metrics |
| **URI** | Uniform Resource Identifier | Web |
| **URL** | Uniform Resource Locator | Web |
| **VRAM** | Video RAM | Hardware |
| **YAML** | YAML Ain't Markup Language | Data Format |

## Common Metrics and Units

| Metric | Unit | Description |
|--------|------|-------------|
| **Latency** | ms (milliseconds) | Response time |
| **Throughput** | tokens/sec | Processing rate |
| **Memory** | GB/TB | RAM or storage capacity |
| **GPU Memory** | GB | VRAM capacity |
| **Bandwidth** | Gbps | Network speed |
| **Concurrency** | requests | Simultaneous requests |
| **Utilization** | % | Resource usage percentage |
| **QPS/RPS** | queries/sec | Request rate |

## Model Size Classifications

| Category | Parameter Range | Memory Required | Example Models |
|----------|----------------|-----------------|----------------|
| **Small** | < 3B | 8-16 GB | GPT-2, BERT |
| **Medium** | 3B - 7B | 16-32 GB | Llama-2-7B, Mistral-7B |
| **Large** | 7B - 30B | 32-128 GB | Llama-2-13B, GPT-J |
| **Very Large** | 30B - 70B | 128-320 GB | Llama-2-70B, Falcon-40B |
| **Massive** | > 70B | 320+ GB | GPT-4, PaLM, Claude |

## Performance Benchmarks Reference

| Metric | Traditional | vLLM | Improvement |
|--------|------------|------|-------------|
| **Throughput** | ~150 tokens/s | ~3,500 tokens/s | 24x |
| **GPU Utilization** | 40-50% | 90-95% | 2x |
| **Memory Efficiency** | 40% | 96% | 2.4x |
| **Latency (P99)** | 2000ms | 400ms | 5x |

---

This glossary provides comprehensive definitions for technical terms, acronyms, and concepts used throughout the LLM inference documentation. For more detailed explanations of specific topics, refer to the corresponding documentation sections.