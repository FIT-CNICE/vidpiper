---BEGIN MARP DECK---
marp-theme: "proposal"
title: "NVIDIA's AI Compute Fabric: Paving the Way for Next-Gen AI"
subtitle: "A Deep Dive into Networking, Storage, and Optimization Strategies"
taxonomy: "AI Infrastructure > Networking > Compute Fabric"
---
## Introduction: NVIDIA's Vision for AI Computing

![width:500px](screenshots/scene_1.jpg)

*   **NVIDIA at GTC 2025 China Network Special Session**: Introducing the "NVIDIA Creates AI Computing Network"
*   **Key Concepts**: Quantum InfiniBand, SuperNICs, Agentic AI, Distributed AI, AI factories
*   **The Core Message**:  Building a new AI computing network, supporting AI development and innovation across industries.

---
## Pave to AI Compute Fabric: A Paradigm Shift

![width:500px](screenshots/scene_2.jpg)

*   **Initiative**: NVIDIA's "Pave To AI Compute Fabric" (March 19, 2025)
*   **LLMs as the Driving Force**: Top-tier LLM providers require powerful computing infrastructure.
*   **Mixture of Experts (MoE) Models**:
    *   **Larger Training Scales**
    *   **Enhanced Inference Capabilities**
    *   **Improved Recommendation Systems**
*   **Data Center as a Unified Computer**: Transitioning from separate machines to a cohesive computing entity.
*   **NVIDIA's End-to-End Solution**:
    *   **Hardware Layer**: GPUs, CPUs, DPUs, Networking
    *   **Platform Layer**: DGX, HGX, Omniverse, AGIX
    *   **Software Layer**: Libraries for Communication, Computation, Domain-Specific Acceleration

---
## NVIDIA's Networking Infrastructure for AI Computing

![width:500px](screenshots/scene_3.jpg)

*   **Data Center as a Unified Compute Fabric**: Transitioning from components to a unified system.
*   **Scale-Out Networks (East-West Communication)**:
    *   InfiniBand Technology:
        *   Quantum InfiniBand Switch
        *   ConnectX SuperNIC
    *   AI Ethernet Technology:
        *   SPECTRUM Ethernet AI Switch
        *   BLUEFIELD SuperNIC
*   **Scale-Up Networks (Within Node Communication)**:
    *   NVLINK Technology:
        *   NVLINK Switch
        *   BLUEFIELD DPU
*   **End-to-End Platform**:  Hardware, Platform Integration, Application Frameworks

---
## LLM Compute and Communication Profiling

![width:500px](screenshots/scene_4.jpg)

*   **Communication Patterns in LLM Training**: Visualization with time-based traces.
*   **Key Insights**:
    *   **Complete Iteration Cycle**:  Compute and Communication phases.
    *   **Compute Phases**: Computation
    *   **Combined Compute and Communication Phases**: Overlap
    *   **Isolated Communication Patterns**: Communication
*   **Critical Network Characteristics**:
    *   Communications are bursty
    *   Average bandwidth utilization is not a good network criteria
*   **Transformations in LLM Training**: Data, Tensor, Pipeline, and Expert Transformations
*   **Collective Communication Operations**: AllReduce, ReduceScatter, AllGather
    *   Scale-Up networks (NVLink) for critical operations.
    *   Scale-Out networks for other operations.

---
## LLM Compute and Communication Optimization Strategies

![width:500px](screenshots/scene_5.jpg)

*   **Three Critical Phases of LLM Training Iterations**:
    *   Compute-Dominant Phase (left): Minimal communication.
    *   Compute + Communication Overlap Phase (middle): Optimization Opportunity.
    *   Exposed Communications Phase (right):  Communication dominates.
*   **Expert Communications in MoE Models**: Expert routing communications are challenging.
*   **Key Insight**:  Communication is bursty and average bandwidth isn't a good metric
*   **Optimization Strategy**:  Maximize computation, overlap communication, and minimize communication-only phases.

---
## NVLink5 Compute Fabric: Next-Generation GPU Interconnect

![width:500px](screenshots/scene_6.jpg)

*   **NVLink5 Compute Fabric**:  "New Era in GPU Interconnect" for AI workloads.
*   **Key Technical Specifications**:
    *   GPU Connection Capacity: Up to 7.2 Tbps (18 ports at 400 Gbps).
    *   NVSwitch ASIC Performance: Up to 28.8 Tbps (72 ports at 400 Gbps).
*   **Advanced Network Features**:
    *   SHARP Technology Integration: Collective communication offloaded to switches.
    *   Multicast Acceleration: Efficient one-to-many data distribution.
*   **Scalable Network Architecture**:  Handles bursty communication.

---
## Quantum-X800 InfiniBand Compute Fabric: High-Performance AI Network

![width:500px](screenshots/scene_7.jpg)

*   **Quantum-X800 InfiniBand Compute Fabric**: NVIDIA's second AI-dedicated networking product.
*   **Key Technical Specifications**:
    *   Network Bandwidth: 115 Tbps (5× increase).
    *   In-Network Computing: SHARP v4 technology (14.4 TFlops, 9× performance improvement).
    *   Advanced Data Types: FP8 and BF8.
*   **Network Architecture Components**:
    *   Quantum-X800 Switch: 144 × 800G ports, SHARP v4, adaptive routing.
    *   ConnectX-8 SuperNIC: PCIe Gen 6 interface, multi-host support.
*   **Intelligent Network Management**: Adaptive routing, congestion control, power management.
*   **Performance Visualization**: Demonstrates SHARP All-Reduce performance.

---
## Quantum-X800 InfiniBand Compute Fabric: Advanced Networking

![width:500px](screenshots/scene_8.jpg)

*   **ConnectX-8 SuperNIC Features**: Integration with NVIDIA's complete networking solution.
*   **PCIe 6.0 Integration Capabilities**:
    *   48-lane PCIe 6.0 switch.
    *   Direct connections between GPUs and networking hardware.
    *   Frees GPUs from server's PCIe version.
*   **Cross-Generation Compatibility**:
    *   ConnectX-8 can connect to PCIe 6.0 GPUs in older PCIe 5.0 infrastructure.
    *   Provides 800G bandwidth.
*   **Multi-Host and Socket Direct Technology**:
    *   Can share 800G bandwidth across multiple processors or GPUs.
*   **Expanding Beyond HPC to Enterprise AI**:
    *   Enhancing Ethernet for AI workloads in enterprise settings.

---
## Spectrum-X: The First Ethernet Network Designed for AI

![width:500px](screenshots/scene_9.jpg)

*   **Spectrum-X Ethernet**: Specifically designed for AI workloads.
*   **Comparison**:
    *   Traditional Ethernet (Hyperscale Clouds): Loose applications, TCP, Low bandwidth, High jitter, Average multi-pathing.
    *   Spectrum-X Ethernet (Generative AI):  Tightly coupled, RoCE, High bandwidth, Low jitter, Bursty network, predictable performance.
*   **Technical Innovations**:
    *   Dynamic Routing Technology: Improves communication efficiency.
    *   Advanced Congestion Control: Consistent performance for each workload.
    *   Performance Isolation: Predictable performance in multi-tenant environments.
*   **Significance for Enterprise AI**: Adapting HPC technologies to enterprise environments.
*   **The Network Defines the Data Center**: Network architecture is a critical factor.

---
## AI Storage Optimization Use Cases

![width:500px](screenshots/scene_10.jpg)

*   **Storage Considerations for AI Model Lifecycle**: Storage is a critical part of AI performance.
*   **Key Storage Challenges for AI Factories**:
    *   Multi-modal Data Fetching
    *   Checkpointing
    *   Inference with RAG
*   **Storage Network Performance Across the AI Lifecycle**:
    *   Data Ingest
    *   Training
    *   Fine Tuning
    *   Inference
*   **Performance Implications**:  Optimizing "north-south" storage traffic can deliver up to 60% better performance.

---
## AI Storage Optimization Use Cases: Foundations for High-Performance AI

![width:500px](screenshots/scene_11.jpg)

*   **Storage Requirements Across the AI Workflow**:
    *   Multi-modal Data Fetching
    *   Checkpointing
    *   Inference with RAG
*   **Critical Storage Network Performance**:
    *   Data Ingest
    *   Training
    *   Fine Tuning
    *   Inference
*   **Checkpointing and Network Implications**:
    *   Terabytes of data.
    *   Fast transfer to remote storage.
    *   Fast retrieval for resuming training.
    *   Must operate without interference.
*   **Inference Performance and User Experience**:
    *   Latency-sensitive.
    *   Fast data retrieval is critical for quick responses.

---
## NVIDIA Spectrum-X: Enhanced Storage Performance

![width:500px](screenshots/scene_12.jpg)

*   **Spectrum-X Storage Performance Benefits**:
    *   48% higher storage bandwidth per GPU.
    *   1.2x higher performance in "noisy neighbor" scenarios.
*   **Technical Architecture**:
    *   NVIDIA Spectrum-4 Ethernet switches.
    *   NVIDIA BlueField-3 DPUs.
    *   Software stack (NVIDIA networking software, Cumulus/SONiC).
*   **Practical Performance Improvements**:
    *   48% storage performance improvement.
    *   Performance isolation, improving performance by 20%.
*   **Industry Adoption**:  DDN, VAST Data, WEKA.

---
## NVIDIA AIR: Digital Twin Technology for AI Networks

![width:500px](screenshots/scene_13.jpg)

*   **Overview of NVIDIA AIR**: Digital twin platform for AI data center networks.
*   **Key Components**: Spectrum-X, MGX, NVIDIA LINK switch.
*   **Benefits of Digital Twin Technology**:
    *   Pre-deployment configuration and testing.
    *   Risk-free maintenance planning.
    *   Parameter transfer.
*   **Real-World Application**:  XAI's Large-Scale Deployment (100,000 NVIDIA GPUs in 19 days).

---
## Spectrum-X 100K GPU AI Supercomputer: Network Performance

![width:500px](screenshots/scene_14.jpg)

*   **Key Performance Metrics**:
    *   Load Balancing: 1.6× higher effective bandwidth.
    *   Tail Latency: 1.3× improvement in collective communications.
    *   Noise Isolation: 2.2× higher all-reduce bandwidth.
    *   Resilient Performance: 1.3× higher all-to-all bandwidth.
    *   Telemetry: 1,000× faster network monitoring.
    *   Programmability:  Built-in network computing resources.
*   **System Architecture**: 2,800 Spectrum-X switches, 100,000 BlueField-3 SuperNICs, 300,000 LinkX cables.
*   **Build Timeline**: Built in 122 days, from rack installation to training in 19 days.
*   **Advanced Network Technologies**: Dynamic routing, flow control, performance isolation, NVIDIA AIR.

---
## Quantum-X800: Advanced Co-packaged Optics

![width:500px](screenshots/scene_15.jpg)

*   **Co-packaged Optics Technology**: Addresses power consumption in AI data centers.
*   **Key Technical Benefits**:
    *   Significant Power Reduction: Reduced from 17W to 4W per port.
    *   Scale Impact:  >1,000W savings per switch.
    *   TCO Savings: Reduced operational costs.
    *   Enhanced Reliability.
*   **Quantum-X800 Switch Specifications**: 144 ports at 800Gb/s, liquid-cooled, 4U platform.
*   **Strategic Importance**:  High-performance, low-power networks, freeing power for GPUs.
*   **Expected Introduction**: End of the year.

---
## NVIDIA DOCA China Developer Community

![width:500px](screenshots/scene_16.jpg)

*   **NVIDIA DOCA Developer Community**: Maximizing performance of NVIDIA networking hardware.
    *   Community Membership.
    *   Software Download.
*   **DOCA Platform Significance**: DOCA (Data Center-on-a-Chip Architecture).
    *   Enables developers.
    *   SuperNICs, DPUs.
*   **AI Data Center Network Optimization**:
    *   Balance compute and storage.
    *   Scale-out optimization.
    *   Power efficiency (CPU offloading).
    *   AI Ethernet (enhanced security and performance).
*   **Closing Remarks**:  Panel discussion moderated by Feng Gaofeng.

---
## AI Computing Center Network Panel Discussion Introduction

![width:500px](screenshots/scene_17.jpg)

*   **Computing Center Evolution and Networking Challenges**: Massive scaling creates fundamental networking challenges.
*   **Introduction of Dennis Cai**: Vice President of R&D at Alibaba Cloud Intelligence Group.
*   **Large AI Models and Infrastructure Requirements**:
    *   Alibaba's AI models.
    *   High-performance networking enables large-scale systems.
*   **The Three Pillars of AI Development**: Algorithms, Computing Power, Data.
*   **Training Cluster Evolution**: From hundreds to 100,000+ GPUs, and now massive clusters.

---
## AI Algorithm Innovation and Network Infrastructure Evolution

![width:500px](screenshots/scene_18.jpg)

*   **Shift in AI Model Development Focus**:
    *   From Scale to Reasoning:  Multi-step, hierarchical, reinforcement learning, chain-of-thought processing.
*   **Model Training Evolution**:
    *   Pre-training.
    *   Post-training.
    *   Inference.
*   **Network Infrastructure Development Phases**:
    *   Foundation Phase (2022-mid 2023):
        *   Networking Innovation.
    *   Stabilization Phase (mid 2023-mid 2024):
        *   Commitment to specific approaches
        *   thousands of GPUs.
    *   Current Evolution Phase (mid 2024-present):
        *   Inference workload dominant.
        *   Multi-level and integrated inference approaches.
        *   Cost-effective high-performance networks.
*   **Scale Challenges for the Future**:
    *   Foundation model training continues.
    *   Cluster sizes increasing from thousands to 100,000+ GPUs (5-10x increase).
    *   Performance consistency and reliability.

---
## GPU Network Architecture Debates and AI Computing Infrastructure

![width:500px](screenshots/scene_19.jpg)

*   **Key Industry Debates**:
    *   Network Scale Expansion.
    *   Architecture Approaches (proprietary vs. open).
    *   Protocol Selection (proprietary, Ethernet, or new).
    *   Scale Limitations.
*   **Fundamental Differences**:  Synchronized communication, "Bucket Effect", system-wide vulnerability.
*   **AI Computing as a Complex Systems Engineering Challenge**: A holistic systems approach to bottlenecks.

---
## Predictable Network Architecture for AI Computing

![width:500px](screenshots/scene_20.jpg)

*   **Alibaba Cloud's "Predictable Network"**:  New design with tight coordination.
    *   Experience with RDMA since late 2019.
    *   HPCC flow control, RDM network protocols, WE Fabric architecture.
*   **HPCC 7.0: Purpose-Built for AI GPU Clusters**:
    *   Two-Network Separation (Front-end/Back-end Split): Simplifies GPU interconnect.
    *   Physical Network Topology and Multi-Path Design:  Multi-path networking.

---
## Network Optimization for Large-Scale AI Clusters

![width:500px](screenshots/scene_21.jpg)

*   **RDMA Traffic Distribution Techniques**:
    *   White-Box RDMI Solution: Distribute network traffic uniformly.
    *   Flow Control System (HPCC): Without lossless networks.
*   **Communication Library Innovations**: Bridge between networking and frameworks.
*   **HPCC 7.0 Architecture Unique Features**:
    *   Dual Network Card Uplinks with Multiple Planes: Enhances stability.
*   **Current Focus Areas**: Stability and performance of ultra-large-scale training clusters.
*   **Focus**: Large cluster stability and performance optimization for inference clusters.

---
## Challenges in Large-Scale AI Training: System Stability and Fault Tolerance

![width:500px](screenshots/scene_22.jpg)

*   **The Scaling Problem: Meta's LLaMA 3.1 Case Study**:
    *   16,000 GPU card cluster: training restarts every 2-3 hours.
    *   100,000 GPU card cluster: restarts every 30 minutes.
*   **Multi-Pronged Approach to Stability**:
    *   Hardware Improvements
    *   Architectural Enhancements:  HPCC 7.0's dual-plane to four network planes.
    *   Rapid Fault Detection System:
*   **Future Directions in Training and Inference**:
    *   Efficiency and architecture
    *   Inference is becoming main driver.

---
## Challenges in AI Inference Systems and Network Architecture

![width:500px](screenshots/scene_23.jpg)

*   **Key Differences Between Training and Inference Clusters**:
    *   Traffic Pattern Differences: Latency sensitive.
    *   Communication Methods: Dynamic, Incast.
*   **Network Performance Challenges**: Heterogeneous hardware, transport protocols (GCP vs. RDMA).
*   **Deeper Hardware Optimizations**:  Reduce GPU SM resource consumption.
*   **Inference Systems**: Require fundamentally different network architectures and optimization.

---
## Technical Interview on ByteDance's AI Computing Infrastructure

![width:500px](screenshots/scene_24.jpg)

*   **Speaker Introduction**: Bingshan, ByteDance, head of network and data center networking.
*   **Topic**: ByteDance's networking for AI computing, comparisons to conventional internet data center networks.
*   **ByteDance's Approach**: Implementing AI technologies in search, recommendation, and real-time reporting.
*   **Investment in Innovations**: Deep in these areas and ecosystem.

---
## ByteDance's Systematic Approach to AI Computing Infrastructure

![width:500px](screenshots/scene_25.jpg)

*   **Integration of Systems and Teams**: System capability across the company.
    *   S4 systems.
    *   APA4 systems.
    *   SMA4 systems.
*   **Optimizing Computing Power Efficiency**:
    *   "Train Time Ratio" (TTR).
    *   "MFU" (Model FLOPs Utilization).
*   **Cross-Departmental Collaboration**:
    *   Infrastructure teams.
    *   Domain-specific technical teams.
    *   Resource integration specialists.
    *   Research and development coordination.
*   **Technical Optimization Process**:
    *   Model decomposition techniques.
    *   Optimizing the "Trans" framework.
    *   Network team system optimizations.

---
## Collaborative Innovation in AI Infrastructure for Optimizing Performance

![width:500px](screenshots/scene_26.jpg)

*   **Multi-faceted Approach to Model Optimization**:
    *   Model decomposition techniques.
    *   Network design integration.
    *   Training optimizations.
*   **Resource Efficiency and Network Fundamentals**: Efficient data transmission, reducing latency, maximizing throughput.
*   **Context-Specific Innovations**: Different requirements for training vs. inference.
*   **Technical Integration Points**: Congestion control algorithms, Protocol design, Network fault coordination mechanisms.

---
## Data Center Innovation for Large-Scale AI Infrastructure

![width:500px](screenshots/scene_27.jpg)

*   **Revolutionary Data Center Design**:
    *   Unprecedented scale (400 MW).
    *   Purpose-built for AI.
    *   Power optimization.
*   **Technical Advantages**: Industry-leading in physical scale and operational efficiency.
*   **Comprehensive Systems Integration**:
    *   Holistic design philosophy.
    *   Standardized yet flexible architecture.
    *   Sensitivity to operational factors.

---
## Network Architecture Considerations for AI Computing Infrastructure

![width:500px](screenshots/scene_28.jpg)

*   **Optimized Network Design**:
    *   Network bandwidth target (400 Gbps).
    *   Network reach (2 kilometers).
    *   Multi-layer approach (3-layer architecture).
*   **Integrated Network and Application Design**:
    *   Traffic pattern awareness.
    *   Model parallelism considerations.
    *   Data parallel operations ("reverse gradients").
*   **Collaborative Design Approach**:
    *   Joint optimization.
    *   Hierarchical planning.

---
## Cost-Effectiveness in AI Network Design

![width:500px](screenshots/scene_29.jpg)

*   **Optimized Network Architecture Decisions**:
    *   Three-tier network architecture.
    *   Strategic bandwidth allocation ("table-linked pen" design).
    *   Cost-performance balance.
*   **Shifting Industry Priorities**:
    *   Last year's approach (performance over cost).
    *   Current approach (sustainability and cost-effectiveness).
    *   Collaborative optimization.
*   **Future Optimization Directions**:
    *   Reducing deployment complications.
    *   Fault detection and recovery.
    *   Avoiding common problems.

---
## Navular Protocol: Innovative Network Solutions for AI Workloads

![width:500px](screenshots/scene_30.jpg)

*   **Protocol Design Innovations**:
    *   Scenario-aware design.
    *   Problem avoidance.
    *   Industry collaboration.
*   **Rocket V2 Protocol Context**:  Well-regarded, but not designed for AI workloads.
*   **Technical Innovations in Navular**:
    *   Multipath optimization.
    *   SACP support.
    *   Inference-specific enhancements.
*   **Workload-Specific Considerations**:
    *   Training vs. inference distinctions.
    *   Latency sensitivity.
    *   Practical implementation.

---
## Network Solutions for Large-Scale AI Clusters

![width:500px](screenshots/scene_31.jpg)

*   **Network Attack Patterns and Overhead**: Attack overhead requires resource-intensive responses.
*   **Multi-Layered Network Solutions**: AR technologies, load balancing (GLB and Paxbury), network standardization.
*   **Stability in Large Model Clusters**:
    *   "Short board effects."
    *   Failure frequency: "a few to ten times per day."
    *   Resilience design.
*   **Practical Implementation Approach**: Layered solutions, performance stability, standards-based.

---
## Fault Tolerance and Recovery Strategies in Network Design

![width:500px](screenshots/scene_32.jpg)

*   **Failure Detection and Response Protocols**:
    *   Failure rates (10% of components).
    *   Detection mechanisms.
    *   Response time optimization.
*   **Recovery Automation Strategy**:
    *   Rapid fault diagnosis.
    *   Tactical recovery points.
    *   Future improvements (near-instantaneous failover).
*   **Network Architecture Advantages**: Single-plane design, software-based recovery.
*   **Customer-Facing Solutions**: Software-based recovery, technology innovations.

---
## Network Innovations for AI System Reliability and Performance

![width:500px](screenshots/scene_33.jpg)

*   **Proactive Network Monitoring and Notifications**:
    *   User-centric notifications.
    *   Partial degradation management.
    *   Business continuity focus.
*   **Granular Network Monitoring Evolution**:
    *   From basic connectivity to detailed analytics.
    *   Fine-grained measurement precision.
    *   Data-driven problem resolution.
*   **East-West vs. North-South Network Communication**: Different requirements for AI training.
*   **Storage Communication Challenges**: Industry focus imbalance.

---
## VPC Networking Optimization for AI Workloads

![width:500px](screenshots/scene_34.jpg)

*   **Current Network Pressure Distribution**:
    *   East-West traffic (GPU-to-GPU).
    *   North-South traffic (compute-to-storage).
    *   Hardware allocation.
*   **VPC Optimization Goals for Version 2.0**: Improved coordination between compute nodes and storage.
*   **Current Approach vs. Future Possibilities**:
    *   Prioritizing VPC performance for specific use cases.
    *   Potential future integration.
*   **Accelerated Network Innovation**:
    *   Multiple improvement points.
    *   Significant acceleration of development.

---
## Introduction of Baidu's Distinguished Architect

![width:500px](screenshots/scene_35.jpg)

*   **Speaker Introduction**: Zheng Ran, Distinguished Architect at Baidu.
*   **Baidu's AI Computing Center**:
    *   AI-specific computing requirements.
    *   Technical infrastructure challenges.
    *   Intelligent Computing Center.
*   **Key Technical Challenges**: Different design considerations than traditional data centers.

---
## The Exponential Growth in AI Computing Infrastructure

![width:500px](screenshots/scene_36.jpg)

*   **Computing Power Requirements**:
    *   Exponential Growth in Parameters.
    *   Two-Pronged Advancement in Computing Hardware.
    *   Computing Efficiency.
*   **Storage and Networking Challenges**:
    *   Storage Requirements (Petabytes to Exabytes)
    *   Networking Infrastructure (Tens or hundreds of gigabytes per second, ultra-low latency).
*   **Baidu's Response**: All-flash storage architectures, high-performance parallel systems.

---
## Advanced Networking for AI Computing Centers

![width:500px](screenshots/scene_37.jpg)

*   **Networking Infrastructure**: High-speed networks, network scaling challenges, rack-level networking.
*   **Thermal Management Innovation**: Power density, traditional cooling limitations, advanced cooling solutions.
*   **Baidu's Network Technology Focus**:
    *   Matching Network Growth to Computing Growth.
    *   Holistic Infrastructure Approach.

---
## Baidu's HPN Network Technology Innovations

![width:500px](screenshots/scene_38.jpg)

*   **Network Architecture Highlights**:
    *   Massive Scale Deployment (51.2T switch chips, 400G, three-layer CLOS).
    *   Multi-Rail Design.
    *   MOE Optimization (A2A communication).
*   **Network Control Systems**:
    *   Dynamic Load Balancing.
    *   Fine-Tuned Traffic Management.
*   **Network Stability Features**:
    *   Fault Tolerance.
    *   Light-Speed Fault Recovery.
*   **Monitoring Capabilities**: Ultra-High Precision Monitoring, Switch Machine Detection.
*   **AI Network Evolution Timeline**: From 2 to 16 card clusters to 10M cards.

---
## Baidu's HPN Network Stability and Traffic Optimization

![width:500px](screenshots/scene_39.jpg)

*   **Advanced Traffic Management**:
    *   Internal POD Routing.
    *   Flow Balancing Technology (reduces ECN).
*   **Network Stability Enhancements**:
    *   Fault Isolation (BCCL reliability by 20%).
    *   Real-time Recovery.
    *   Smart Traffic Management.
*   **Monitoring Capabilities**:
    *   Ultra-Precise Monitoring (switch and network card levels).
    *   PingMesh Detection.

---
## Baidu HPN Network Technical Innovations and Roadmap

![width:500px](screenshots/scene_40.jpg)

*   **Network Architecture Highlights**: Massive Scale, multi-plane design.
*   **Dynamic Traffic Management**: Adaptive routing, dynamic load balancing.
*   **Stability Enhancements**: Fault Management, mutual fault recovery, exchange machine management.
*   **Monitoring Capabilities**: High-precision, exchange machine diagnosis.
*   **Development Timeline and Roadmap**: From 2021 to 2024 (2T-16 card clusters to 10M cards).

---
## Baidu HPN Network Technical Stability and Observability Features

![width:500px](screenshots/scene_41.jpg)

*   **Network Stability Enhancements**:
    *   Fault Detection and Isolation.
    *   Mutual Fault Recovery.
    *   Exchange Management During Failures.
*   **Advanced Monitoring Capabilities**:
    *   High-Precision Monitoring.
    *   PingMesh Diagnostics.

---
## Baidu's Open White Box Switching and Network Innovation

![width:500px](screenshots/scene_42.jpg)

*   **Baidu's Custom Switch Development Journey**:
    *   Long-term Experience.
    *   Open White Box Approach.
*   **Strategic Benefits of Custom Switching**:
    *   First-mover Advantage.
    *   Enhanced Control (specialized software, protocols).
*   **Industry Collaboration**: Partnership with NVIDIA.
*   **AI-Driven Network Evolution**: Part of a complete solution.

---
## Transforming AI Infrastructure

![width:500px](screenshots/scene_43.jpg)

*   **Expanding AI Resource Capabilities Beyond Internal Use**:
    *   Cloud Service Orientation.
    *   Technology Transfer.
*   **Technical Challenges for AI Cloud Services**:
    *   Computing Resource Diversification.
    *   Resource Management Systems.
    *   Network Architecture Evolution.
*   **Strategic Significance**:  AI infrastructure as a service.

---
## Evolution of Network Architecture for AI Workloads

![width:500px](screenshots/scene_44.jpg)

*   **Addressing Network Architecture Limitations**:  Current systems optimized for either training or inference.
*   **The Vision for AI-Optimized Networks**:
    *   Software-Defined Infrastructure.
    *   Enhanced Flexibility.
    *   Integration of IT Network Technologies.
    *   Unified Network Definition.
*   **Data Storage Considerations for AI Models**: High data quality, unstructured data challenges, AI-aware storage systems.
*   **Comprehensive AI Computation Lifecycle**:  Data preprocessing, efficient data loading, storage-compute integration.

---
## Conclusion of Technical Discussion on Data Organization

![width:500px](screenshots/scene_45.jpg)

*   **Data Organization Improvements for Business Intelligence**:
    *   Enhanced BX Data Attributes.
    *   Stage-Specific Data Standards.
    *   Multi-Dimensional Analysis.
    *   Performance-Driven Approach.
*   **Three-Pronged Improvement Strategy**:
    *   Computational Power Evolution.
    *   Integration of Systems.
    *   Advanced Data Organization.
*   **Transition to Next Speaker**:  Kong Yu from Kuaishou.

---
## Kuaishou's Recommendation System Architecture

![width:500px](screenshots/scene_46.jpg)

*   **Overview**:  Kuaishou's recommendation system for 400M DAU.
*   **The Bipartite Graph Approach**: Connects consumers (users) and suppliers (content).
*   **Recommendation System Pipeline (Within 1 second)**:
    *   Content Index.
    *   Retrieval Stage (TDM, GNN, Policy Rules).
    *   Ranking Stage (SIM, LTR, EnsembleSort).
    *   Reranking Stage (Multi Business Objectives).
*   **Technical Challenges**: Balance multiple factors, manage resources, and optimize objectives.

---
## Increasing Computational Density in Recommendation Systems

![width:500px](screenshots/scene_47.jpg)

*   **Novel Approach to CPU Workload Distribution**: Addressing CPU bottlenecks.
*   **Traditional vs. New Approach**:
    *   Offload CPU to remote CPU servers (increases network overhead).
    *   Offload CPU to local resources (GPUs/DPUs).
*   **Technical Implementation**: Moving tasks to local GPUs/DPUs.

---
## Optimizing Computational Efficiency by Offloading CPU Workloads

![width:500px](screenshots/scene_48.jpg)

*   **System Architecture Optimization**:  Distributing workloads across processors.
*   **Technical Implementation Details**:
    *   CPU workload.
    *   GPU workload.
    *   DPU workload.
*   **Communication Optimization**: DMA for low-latency communication.
*   **Performance Results**:  1.65× speedup (65% improvement).

---
## Workflow Optimization Through CPU Workload Offloading

![width:500px](screenshots/scene_49.jpg)

*   **System Workflow Architecture**: Optimized workflow for a 65% improvement.
    *   CPU Workload.
    *   Local Processing.
    *   DPU Workload.
    *   GPU Workload.
    *   Result Processing.
*   **Technical Implementation Benefits**: CPU bottlenecks eliminated, DPU handles data fetching, GPU for processing.
*   **Communication Optimization**: DMA and RDMA for efficient data flow.

---
## Challenges of Traditional Storage Architectures for AI

![width:500px](screenshots/scene_50.jpg)

*   **The Limitations of Coupled Computing and Storage**: Inadequate for modern AI workloads.
*   **Three Critical Limitations**:
    *   Performance Scaling Constraints
    *   Low Resource Utilization
    *   High Maintenance Complexity
*   **Moving Toward Disaggregated Architecture**: Separating computing and storage.

---
## Disaggregated Computing and Storage as a Key Direction

![width:500px](screenshots/scene_51.jpg)

*   **The Benefits of Separating Computing and Storage**:
    *   Enhanced Scalability.
    *   Optimized Resource Allocation.
    *   Simplified System Maintenance.
    *   Improved Security.
*   **SIF Storage System**: Selected for its high performance and scalability.
*   **New Disaggregated Approach**: Using NVIDIA BlueField DPUs and Samsung's PCIe SSDs.

---
## Limitations of Traditional CEPH Deployment Architecture

![width:500px](screenshots/scene_52.jpg)

*   **Traditional CEPH Deployment Challenges**:
    *   High TCO (Three-copy deployment).
    *   Inefficient SSD Performance Utilization.
    *   Limited Flexibility and Scalability.
    *   Low Resource Utilization.
*   **Architecture**: Illustrates bottlenecks and storage inefficiencies.

---
## BlueField-3 DPU Performance for CEPH Storage Optimization

![width:500px](screenshots/scene_53.jpg)

*   **BlueField-3 DPU Capabilities**: Offloading CEPH workloads.
*   **Technical Benefits**:
    *   ARM Computing Power.
    *   NVMe SSD Performance (S2IOV).
    *   Optimized Resource Utilization.
    *   Network Acceleration.
*   **Technical Impact**: Increased IOPS and a more flexible system.

---
## CEPH Cluster Performance Testing Results

![width:500px](screenshots/scene_54.jpg)

*   **Impressive Performance Breakthrough**:
    *   Record-Breaking Performance (45 Gbps).
    *   3x Performance Improvement.
    *   Future Scalability (PCIe Gen5 SSDs).
*   **Technical Implementation Details**: Architecture.
*   **Performance Comparison**: 3x improvement.

---
## Storage System Enhancements for High-Performance Computing

![width:500px](screenshots/scene_55.jpg)

*   **Advanced Storage Architecture for Multiple Systems**: Across HPC file systems.
*   **Technical Implementation Details**:
    *   DPU vs CPU Offloading.
    *   NVMe SSD Performance.
    *   SR-IOV Technology.
    *   PCIe Gen5 SSDs.
*   