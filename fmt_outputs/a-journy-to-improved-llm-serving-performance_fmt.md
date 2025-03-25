---
marp-theme: proposal
title: "A Journey to Improved LLM Serving Performance"
subtitle: "SAP's AI Foundation and Deployment Strategies"
taxonomy: "LLM; AI; Deployment; Performance; SAP"
---
## Introduction: SAP's AI Foundation

![width:500px](https://raw.githubusercontent.com/fit-sizhe/dwhelper/main/gtc/sum_outputs/a-journy-to-improved-llm-serving-performance_output/screenshots/scene_1.jpg)

*   **SAP's Vision:** Integrating AI across its product portfolio.
*   **Layered Approach:** "Jule" (co-pilot) at the top, integrated with business systems. AI foundation built on the Business Technology Platform (BTP).
*   **Generative AI Hub:** One-stop shop for AI developers, providing access to models (Mistral, Meta, OpenAI, Google, Anthropic, custom SAP models). Simplified legal and commercial frameworks.
*   **AI Launchpad:** UI with chat playground, prompt engineering tools, and prompt management.
*   **Technical Focus:** Hosting, scaling, serving LLMs with enhanced throughput, minimal latency, and improving time-to-production. Built on Kubernetes, using "Gardener" for node initialization. Challenges: managing container images (10GB+).

---
## Optimizing LLM Deployment

![width:500px](https://raw.githubusercontent.com/fit-sizhe/dwhelper/main/gtc/sum_outputs/a-journy-to-improved-llm-serving-performance_output/screenshots/scene_2.jpg)

### Reducing Deployment Time

*   **Problem:** Reducing deployment times to improve time to market and development cycles.
*   **Faster Model Downloads:**
    *   Replaced Boto3 with S5CMD (open-source tool) for object store downloads.
    *   Switched to local NVMe SSDs from HDD-backed EBS storage.
    *   Result: Download time for 70B parameter model (~130GB) reduced from 22 minutes to ~3 minutes.
*   **Dedicated Resources:** GPU node worker pools.
*   **Planned Optimizations:** Model sharding, model caching, and image caching.

---
## LLM Inference Performance

![width:500px](https://raw.githubusercontent.com/fit-sizhe/dwhelper/main/gtc/sum_outputs/a-journy-to-improved-llm-serving-performance_output/screenshots/scene_3.jpg)

### Comparing Inference Engines

*   **VLLM:** Open-source, optimized for memory management (KV caching, page retention) and continuous batching.
*   **NIM:** NVIDIA's packaged container solution optimized for specific models.
*   **Benchmark (Llama 3.3 70B):**
    *   NIM outperformed VLLM.
    *   **Peak Throughput:** NIM achieved 8,500+ tokens/second (1.4x VLLM).
    *   **Latency (p99):** NIM showed 1.2x better response times.
    *   **Requests Per Second:** NIM handled 1.3x more requests.
*   **Tensor Parallelism:**
    *   8B parameter model on A100 GPUs (40GB).
    *   2 GPUs: ~1.5x throughput increase.
    *   4 GPUs: ~1.7x throughput increase.
    *   Similar gains in latency.
*   **Key Takeaways:** Inference engine selection is crucial, results vary, organizations should benchmark, optimization depends on priorities.

---
## Time to Production & Hardware Performance

![width:500px](https://raw.githubusercontent.com/fit-sizhe/dwhelper/main/gtc/sum_outputs/a-journy-to-improved-llm-serving-performance_output/screenshots/scene_4.jpg)

### H200 vs. A100

*   **Test Parameters:** 100 input tokens, 100 output tokens, 500 concurrent requests.
*   **H200 Advantages:**
    *   1.7x better performance.
    *   1.5x better latency.
    *   1.7x more requests per second.
*   **Context:** Benefits depend on use cases, access patterns, and cost.

### Streamlining the Deployment Pipeline

*   **Open-Source Pipeline:** Detailed workflow: Packaging & Deployment, Evaluation, Security, Operations, Deployment.
*   **Proprietary Solutions (NVIDIA):** Ready-to-use containers, faster benchmarking, reduced security overhead, no image management, enterprise support.
*   **Benefit:** Up to 40% reduction in time-to-production.

---
## Key Takeaways

![width:500px](https://raw.githubusercontent.com/fit-sizhe/dwhelper/main/gtc/sum_outputs/a-journy-to-improved-llm-serving-performance_output/screenshots/scene_5.jpg)

*   **Hardware Performance**: NVIDIA H200 GPUs deliver significant performance improvements over A100s, including better throughput, lower latency, and higher request handling capacity.
*   **Deployment Strategies**: The presentation highlighted a contrast between traditional open-source LLM deployment pipelines, which demand comprehensive steps like packaging, evaluation, and security measures, and NVIDIA's proprietary solutions (like NVDL with TensorRT LLM), which offer pre-optimized containers, potentially slashing deployment time by up to 40%.
*   **Success Factors**: Effective enterprise LLM deployments hinge on scalability, performance (optimizing throughput and latency), and efficiency (reducing time-to-production through streamlined processes). Strategic choices in hardware and deployment approaches are essential for balancing performance, security, and operational efficiency in enterprise AI applications.