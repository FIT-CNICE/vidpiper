---
marp-theme: proposal
title: "Laiye AI Foundry: Technical Overview"
subtitle: "GTC Presentation Summary"
taxonomy: "AI Foundry > Data Processing > PDF Processing"
---
## Laiye AI Foundry: Multimodal PDF Data Processing

![width:500px](Scene_1.jpg)

*   **Laiye AI Foundry Overview**: Introduces data services focusing on extracting information from PDFs.
*   **Challenge**: Extracting valuable data (text, tables, charts, images) from complex PDF documents.
*   **Workflow**: Uses NV-Ingest Blueprint for multimodal data extraction.

---
## Key Technical Components:

*   **Retrieval Pipeline**: Processes user queries using NeMo components (embeddings, vector databases, LLMs).
*   **Ingestion Pipeline**: Processes documents through various extraction stages.

---
## Key Processing Elements:

*   **PDF Parser**: Converts PDF documents into processable formats.
*   **Object Detection**: Identifies and separates visual elements.
*   **Chart/Table Extraction**: Specialized pipelines for structured data.
*   **OCR**: Extracts text from images.
*   **Post-Processing**: Refines extracted data.

---
## Performance Improvements:

*   **Ingestion Speed**: ~3x faster with Laiye enabled (15 pages/sec vs. 4-5 pages/sec).
*   **Retrieval Accuracy**: ~20% improvement with Laiye enabled (80% vs. 60%).
*   **Benefit**: Captures context from PDFs missed by text-only methods.

---
## Laiye AI Foundry: Data Services based on NVIDIA NeMo Curator

![width:500px](Scene_2.jpg)

*   **Focus**: Data processing capabilities built on NVIDIA NeMo Curator.
*   **Objective**: Comprehensive data processing pipeline for AI training.

---
## Data Processing Pipeline Architecture

*   **Data Sources**: Cloud storage, internet, local workstations.
*   **Processing Stages**: Download, extract, clean, filter, deduplicate (exact, fuzzy, semantic), model-based quality filtering, PII removal, blending/shuffling.
*   **AI Model Integration**: LLM NIM, synthetic data generation, NeMo Retriever, LLM/Reward model embeddings, NeMo Curator classifier models.

---
## Performance Improvements

*   **Fuzzy Deduplication**: 16x speedup on an 8TB dataset.
    *   "With Data Curator Off": ~10 hours.
    *   "With Data Curator On": <1 hour.

---
## Key Capabilities:

*   **Multimodal Data Processing**: Handles text, images, video, and other data types.
*   **Synthetic Data Generation**: Creates training data using large language models.
*   **GPU Acceleration**: Uses NVIDIA technologies (DASK, RAPIDS, cuGraph, cuML) for multi-GPU processing.
*   **Privacy Protection**: Tools for PII identification and removal.
*   **Customizable Interfaces**: Allows building custom training data pipelines.
*   **Benefit**: Significantly speeds up data preparation for AI model training.

---
## Laiye AI Foundry: Large Language Model Fine-tuning Process

![width:500px](Scene_3.jpg)

*   **Focus**: Detailed LLM fine-tuning workflow.
*   **Goal**: From raw data to optimized, task-specific models.

---
## Data-Driven Fine-tuning Pipeline

*   **Data Processing Pipeline**: Task descriptions/input data -> cleaning, deduplication, quality filtering -> SFT data pairs -> CoT enhancement -> trainable datasets (instruction-response pairs).

---
## Parameter-Efficient Fine-Tuning (PEFT) Selection

*   **PEFT Methods**: Prompt/Prefix/LoRA/IA³/Adapters.
*   **Structural Features**: Intra/inter-connectivity, insertion forms, parameter adaptation, workspace options.
*   **Task Matching**: Adapts methods to specific tasks (question answering, text generation, classification, summarization).
*   **Enhancements**: Adaptive rank adjustment, adapter diversification.

---
## Benchmarking and Evaluation

*   **Datasets**: Open and customized test datasets.
*   **Evaluation Methods**: Statistical (precision, recall), model-based (similarity), hybrid (F1-score, BLEU-score, Rouge-score).
*   **Goal**: Ensure baseline capabilities while meeting specific business needs.

---
## Advanced Fine-tuning Innovations

*   **ICTO Method**: Combines LoRA, prefix tuning, and adapters; reduces parameter updates, includes a gating mechanism.
*   **SPC Method**: Creates a Mixture-of-Experts (MoE) architecture for prefix networks; improves parameter efficiency and model performance.

---
## Laiye AI Foundry: Comprehensive Architecture Overview

![width:500px](Scene_4.jpg)

*   **Focus**: Overview of the Laiye AI Foundry platform architecture.
*   **Objective**: Efficient, reliable, and flexible generative AI development and deployment.

---
## Layered Technical Architecture

*   **AI Infrastructure (底层基础设施)**: Foundation for AI operations.
*   **KAA Large-Scale Cluster Management System (大规模集群管理系统)**: Optimizes I/O, provides virtualization, manages resource scheduling, offers multi-tenant isolation, usage metering, and dynamic deployment.
*   **MANAS LLMOps Platform (基于图形界面的任务模块化生产力平台)**: Data preprocessing, pre-training, model alignment (SteerLM, DPO, RLHF), model fine-tuning (LoRA, Ptuning, IA³, Adapters).
*   **LIM Large Model Inference Microservice (大模型推理微服务)**: Streamlined model deployment.

---
## Application Scenarios and Services

*   **Industry Applications**: Weather forecasting, biomedical research, finance, e-commerce, education, etc.
*   **Service Categories**: Large model microservices, knowledge library, model security, expansion services.
*   **Key Technology**: NVIDIA AI Enterprise components (DCM cluster management) for GPU cluster management.

---
## Performance and Benefits

*   **Integration**: NVIDIA NEMO for data processing, NEMO framework for customization, NEMO Guardrails for security.
*   **Outcome**: Exceptional concurrent processing, throughput, and low latency for real-world applications.
*   **Goal**: Maximize resource utilization and overall model production capacity.

---
## Laiye High-Performance Training Framework

![width:500px](Scene_5.jpg)

*   **Focus**: High-performance training framework.
*   **Objective**: Optimize NVIDIA's Nemo Megatron training for efficiency.

---
## Framework Components

*   **Parallelism Manager**: Data, Tensor (1D/2D), Sequence, Interleaved Pipeline Parallelism.
*   **Heterogeneous Memory Utilization**: Bandwidth maximizing, memory utilization improvement, Dynamic Memory Swapper.
*   **Activation Memory Optimizer**: Activation Checkpoint, Recomputation, Partition.
*   **Computation Acceleration**: Mixed Precision (FP8, FP16), Graph Optimization, Flash Attention, Triton Layers, Stride Batched GEMM.
*   **Efficiency Analyzers**: Communication, Computation.
*   **LAIYE Efficient Training Monitor**: Monitoring overall performance.

---
## Performance Improvements

*   **Model Flops Utilization (MFU)**: Doubled MFU with Laiye optimization.
    *   Without Laiye optimization: 25% MFU.
    *   With Laiye optimization: 50% MFU.
    *   Example: 128 GPU node cluster

---
## Technical Benefits

*   **Efficient Parallelism**: Maximizes hardware utilization across multi-node, multi-GPU environments.
*   **Memory Optimization**: Manages GPU and system memory, overcomes limitations.
*   **Precision Optimization**: NVIDIA FP8 operations reduce computational demands, and graphic bandwidth.
*   **Computational Graph Optimization**: Reorganizes calculation flows to increase execution efficiency.
*   **Benefit**: Faster training times and better resource utilization.

---
## Laiye AI Foundry - Model Inference Service

![width:500px](Scene_6.jpg)

*   **Focus**: AI model deployment after training.
*   **Objective**: Optimize AI model deployment after the training phase.

---
## Architecture Overview

*   **Memory Node Cluster (top)**: Multiple memory nodes, CPU memory, GPU-Direct connections, KVCache Scheduler, Transformer-based attention computing, Central KVCache Transformer.
*   **Computation Node Cluster (bottom)**: Computation nodes, model parameters, Prefill computing, Non-attention transformer computing, Interleaved pipeline.

---
## Key Optimization Strategies

### 1. Caching Mechanism Optimization
*   **Response Cache**: Stores previous responses.
*   **Radix Tree Implementation**: Reduces lookup times.
*   **Benefit**: Retrieves results directly from cache.

### 2. Inference Process Optimization
*   **Draft Model Fine-tuning**: Achieves 87% acceptance rate, reduces subsequent computation.
*   **Pipeline Task Scheduling**:
    *   Prefill phase (computation-intensive).
    *   Decoding phase (memory-access intensive).
    *   Optimizes hardware resource utilization.

### 3. Hardware Resource Allocation
*   KVCache data on GPUs with better memory price-performance ratio.
*   Intensive computations on GPUs with better computational price-performance ratio.
*   **Benefit**: Maximizes device throughput and maintains inference performance.

---
## Laiye AI Foundry - Large Model Security Services

![width:500px](Scene_7.jpg)

*   **Focus**: Security solutions for large language models.
*   **Key Technology**: NVIDIA NeMo Guardrails as a core protection mechanism.

---
## Security Performance Metrics

*   **OWASP LLM Vulnerabilities Protection**:
    *   Prompt Injection: 45% -> 100% (with full protection).
    *   Insecure Output Handling: Near perfect (100%).
    *   Sensitive Info Disclosure: 85% -> 100%.
    *   Model Theft Protection: 77% -> 100%.
*   **Content Safety Protection**:
    *   Harmful/Violent Content: 100%.
    *   Hate/Harassment: 96% -> 100%.
    *   Profanity: 97% -> 100%.
    *   Security-Confidentiality: 77% -> 100%.

---
## Security Architecture

*   **Input Processing Flow**: User queries pass through multiple security checkpoints and third-party verification.
*   **Security Components**: Content moderation, off-topic detection, RAG enforcement, jailbreak detection, PII detection.
*   **Integration**: NVIDIA NIM.

---
## Key Security Capabilities

*   **Content Monitoring**: Prevents harmful content generation.
*   **Sensitive Data Protection**: Detects and anonymizes PII.
*   **Factual Verification**: Ensures model outputs match verified info.
*   **Attack Prevention**: Blocks attempts to bypass restrictions.
*   **Topic Control**: Maintains response relevance.
*   **Goal**: Meet regulatory compliance in industries like healthcare and finance.

---
## Laiye AI MTP Optimization for DeepSeek Models

![width:500px](Scene_8.jpg)

*   **Focus**: Multi-Token Prediction (MTP) technology for DeepSeek models.
*   **Objective**: Optimize inference performance.

---
## Performance Improvements

*   **Throughput**: Higher with MTP.
    *   Example: 51.12 tokens/sec (MTP) vs. 23.62 (non-MTP) for 512 input/output.
*   **TPOT (Time Per Output Token)**: Lower with MTP.
*   **TTFT (Time To First Token)**: Generally better with MTP.
*   **Example**: For 512 input/output length with batch size 1: TTFT is 181.00 ms vs 1993.39 ms (approximately 11x faster).

---
## Technical Architecture

*   **Main Model**: Core model with transformer blocks and embedding layers.
*   **MTP Modules**: Two parallel modules, specialized output heads.
*   **Cross-Entropy Loss**: Applied at different token positions.
*   **Shared Components**: Elements shared between modules.
*   **RMSNorm Layers**: Normalization in projection paths.

---
## Key Benefits Explained

*   **Higher Throughput**: Predicts multiple tokens simultaneously.
*   **Reduced Latency**: Lowers time to first token.
*   **Improved Batch Processing**: Efficiently handles concurrent requests.
*   **Impact**: Improves response speed.

---
## Laiye AI's Custom Model Development Framework

![width:500px](Scene_9.jpg)

*   **Focus**: Custom model development platform.
*   **Objective**: Efficiently develop customized AI models through a GUI.
*   **Foundation**: Based on NVIDIA AI Enterprise's NeMo Framework.
*   **Platform**: LAIYE MANAS

---
## Key Components of the Workflow

1.  **Data Curation**: Efficient processing of large-scale datasets, generation of synthetic data, and tools for preparing high-quality data.
2.  **Training**: Optimized data loaders, advanced parallelism techniques, memory-optimized processes, and specialized training recipes.
3.  **Alignment**: Support for alignment algorithms like DPO, PPO, RLHF, SteerLM, SFT.
4.  **Customization**: Parameter-efficient methods like LoRA and P-Tuning.

---
## Technical Infrastructure

*   NVIDIA CUDA-Accelerated SDKs
*   TensorRT-LLM (NVIDIA NIM)
*   NeMo Guardrails
*   Community and Nemotron models
*   **User Experience**: Simple UI interface.
*   **Business Value**: Democratizes AI model customization.

---
## Laiye AI Foundry - RAG Knowledge Base Service

![width:500px](Scene_10.jpg)

*   **Focus**: Retrieval-Augmented Generation (RAG) knowledge base service.
*   **Objective**: Intelligent information retrieval and processing.
*   **Architecture**: Hybrid RAG with Graph and Vector RAG.

---
## Query Processing Workflow

1.  **Query Classification**.
2.  **Retrieval**: Query decomposition, rewriting, and hybrid retrieval.
3.  **Reranking**.
4.  **Repacking**.
5.  **Summarizing**.

---
## Data Processing Infrastructure

1.  **Chunking Track**: Semantic chunking, small2big, content embedded.
2.  **Graphing Track**: Node embedding, PageRank, Graph Database.
*   **Benefits**: Combines graph/vector retrieval, supports complex queries, high-speed performance.

---
## Laiye AI Foundry - Management Platform

![width:500px](Scene_11.jpg)

*   **Focus**: Management platform for GPU clusters.
*   **Technology**: NVIDIA BCME-based.

---
## Cluster Deployment and Management Visualization

*   **Visualization**: Grid layout of GPU nodes (green: active, gray: available, red: issues).
*   **Resource Monitoring**: Network, resource utilization, node status.
*   **Performance Profiling**: GPU-specific metrics.

---
## Key Platform Benefits

1.  **Unified Management**: Regardless of cluster size.
2.  **Dynamic Scaling**: Kubernetes integration.
3.  **Policy-Based Allocation**: Maximizes efficiency.
4.  **Cross-Environment Support**: ARM-based edge to CPU servers.
5.  **Detailed Reporting**: Organized by project/application.
*   **Goal**: Reduce administrative overhead, optimize resource utilization.

---
## Laiye AI Foundry - Data Services

![width:500px](Scene_17.jpg)

*   **Focus**: Laiye AI Foundry's data services.
*   **Objective**: Simplify data processing workflows, enhance data value.

---
## Core Data Processing Capabilities

1.  **Unstructured Data Processing**: Handles diverse data formats.
2.  **Multi-Modal Data Processing**: Processes different data types simultaneously.
3.  **Vertical Industry Data Mining**: Extracts info from industry-specific data.
4.  **Self-Labeling and Data Association**: Automates data labeling.

---
## Data Architecture Framework

1.  **Data Sources**: Business applications, enterprise, device logs, external data.
2.  **Data Storage**: Relational and data lakes.
3.  **Synthetic Data**: Generate training data.
    *   **Goal**: More accessible and valuable data for AI applications.

---
## Laiye AI Factory: Healthcare Consultation Implementation Case Study

![width:500px](Scene_18.jpg)

*   **Focus**: Healthcare consultation service.
*   **Implementation**: Comprehensive system architecture.

---
## Healthcare Consultation Workflow Architecture

1.  **Data Processing Layer**:
    *   Chunking.
    *   Relationship Mapping.
    *   Vector and Graph Databases.
2.  **Model Processing Layer**:
    *   Query, consultation classification, search, reranking, repackaging, summarization.
3.  **Evaluation Layer**: Scoring system.

---
## Healthcare Management Workflow

1.  Start → Customer information collection.
2.  Health assessment → Health report generation.
3.  Treatment plan recommendation → Health management solutions.
4.  Effectiveness evaluation → Follow-up health reports.
5.  Health management → Final health reports.
6.  End.

---
## Technical Foundation

*   Body health indicators, healthcare knowledge, medical imaging.
*   Trained through classification, search, reranking, summarization.
    *   **Benefit**: Supports the entire healthcare consultation workflow.

---
## Laiye AI Data Services: Fast Mode & Expert Mode

![width:500px](Scene_19.jpg)

*   **Focus**: Dual approach to data services.
*   **Objective**: Provide services for all technical skill levels.

---
## Fast Mode

*   **Interface**: User-friendly web interface.
*   **Features**: Streamlined upload, direct file management, automated processing, visualization of results, quick deployment.

---
## Expert Mode

*   **Features**: Code examples, GPU resource allocation, custom notebook environments, advanced data processing, text cleaning functionality, and integration with NVIDIA tools.

---
## Key Technical Concepts Explained

*   **Model Fine-Tuning**: Process of adapting pre-trained AI models.
*   **LoRA (Low-Rank Adaptation)**: Adapts large language models by modifying only a small subset of parameters.
*   **PEFT (Parameter-Efficient Fine-Tuning)**: Methods that modify only a fraction of the model parameters.

---
## Platform Benefits

*   Flexibility for all user types.
*   End-to-end solution.
*   Efficient resource utilization.
*   Integrated testing.

---
## Laiye AI Foundry: Products and Services Overview

![width:500px](Scene_20.jpg)

*   **Focus**: Comprehensive product and service offerings.
*   **Objective**: Versatility for enterprise needs.

---
## Two Core Product Lines

1.  **Large-Scale Model Clusters**:
    *   LAIYE AI Foundry.
    *   NVAIE Inside (NVIDIA AI Enterprise integration).
2.  **Enterprise-Grade All-in-One System**:
    *   LAIYE AI Foundry.
    *   NVAIE Inside.

---
## Key Platform Capabilities

*   Extreme adaptability.
    *   Single-machine deployments.
    *   Scalable to thousands of GPUs.
    *   Enterprise toolchain through NVIDIA AI Enterprise.

---
## Laiye AI Foundry - RAG Knowledge Base Service Performance Metrics

![width:500px](Scene_21.jpg)

*   **Focus**: Retrieval-Augmented Generation (RAG) knowledge base service.
*   **Objective**: Focus on performance metrics for fraud detection capabilities.

---
## Accuracy Improvement Metrics

*   **Improvement**: Substantial improvement in fraud transaction detection accuracy.
    *   First bar: 43.23% accuracy.
    *   Second bar: 53.23% accuracy.
    *   Third bar: 85.29% accuracy.

---
## Knowledge Base Management Features

*   Enterprise users can manage knowledge bases through a graphical interface.
*   Process data both online and offline.
*   Store enterprise private data in vector and graph databases.
*   Conduct rapid searches to retrieve knowledge information.
*   Provide precise answers to user queries.

---
## Evaluation Services

*   Ensure that various indicators of the knowledge base meet production requirements.
*   Help maintain quality standards for knowledge retrieval processes.
    *   **Benefit**: Enables effective information retrieval, and fraud detection.

---
## Conclusion of Laiye AI Technical Presentation

![width:500px](Scene_22.jpg)

*   **Focus**: Conclusion of the presentation.
*   **Summary**: RAG (Retrieval-Augmented Generation) knowledge base service and fraud detection system.

---
## Laiye Technology: A One-Stop AI Platform

![width:500px](Scene_23.jpg)

*   **Focus**: Overview of Laiye Technology.
*   **Positioning**: One-stop platform.

---
## Key Technical Offerings and Capabilities

*   **AI Factory**: Foundation for large-scale model solutions.
*   **Blackwell**: Infrastructure for model training and fine-tuning.
*   **NIMS**: Neural information management system.
*   **Omniverse/Robotics**: AI robotics solutions.

---
## Core Value Proposition

*   Cost-effective AI transformation.
*   Large-scale model training and fine-tuning.
*   Intelligent decision-making.

---
## Strategic Vision

*   Mission and Vision.
*   Leading global AI transformation platform.
*   Pioneer in China's intelligent innovation.

---
## Detail-Oriented Takeaways

*   Laiye AI Foundry offers a comprehensive solution for processing multimodal data, including PDFs with text, charts, tables, and images, significantly improving data extraction capabilities.
*   The platform utilizes NVIDIA NeMo Curator to create an end-to-end data processing pipeline for AI training, accelerating data preparation through techniques like fuzzy deduplication and GPU acceleration.
*   Laiye AI's large language model fine-tuning process employs both Quick and Expert modes, accommodating users with different levels of technical expertise, and implements advanced methods like ICTO and SPC for enhanced performance.
*   The Laiye AI Foundry architecture provides a layered system for efficient generative AI development, leveraging NVIDIA AI Enterprise components for cluster management, resource optimization, and secure multi-tenant isolation.
*   The high-performance training framework incorporates multiple parallelism techniques, memory and precision optimization, and computational graph enhancements to increase Model Flops Utilization (MFU) in GPU training environments.
*   Laiye AI's Model Inference Service employs optimized caching mechanisms, pipeline task scheduling, and strategic hardware resource allocation to improve inference performance, particularly in handling DeepSeek models.
*   Laiye AI Foundry provides comprehensive security solutions for large language models (LLMs) using NVIDIA NeMo Guardrails, with improved protection against prompt injection, insecure output handling, and information disclosure.
*   Laiye AI leverages Multi-Token Prediction (MTP) technology to improve inference performance for DeepSeek R1/V3 models, enhancing throughput, reducing latency, and improving batch processing efficiency.
*   The custom model development framework, LAIYE MANAS, offers an end-to-end solution for efficiently developing customized AI models through a graphical user interface, built on the NVIDIA NeMo Framework.
*   The RAG (Retrieval-Augmented Generation) knowledge base service is designed for intelligent information retrieval, with a hybrid architecture and a dual-track data processing system, resulting in significant accuracy improvements in applications like fraud detection.
*   The management platform leverages NVIDIA BCME-based cluster management to provide comprehensive monitoring, unified infrastructure management, and efficient resource scheduling for GPU clusters.
*   The data services offer various capabilities for unstructured and multi-modal data processing, vertical industry data mining, and self-labeling, to streamline the data processing workflows and enhance data value.
*   In healthcare consultation, Laiye AI Factory integrates multiple AI technologies to support the entire healthcare consultation workflow, from customer information collection to treatment recommendations.
*   The Fast Mode and Expert Mode allow Laiye AI Factory to serve both non-technical users who need simple, efficient data processing and technical experts who require deeper customization and programming capabilities.
*   NVIDIA NIM Localization Service for Laiye AI models provides enterprise-ready implementation through rapid adaptation, localized optimization, and a unified interface for easy access to cutting-edge models.