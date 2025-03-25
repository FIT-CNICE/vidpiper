```yaml
---
marp-theme: proposal
title: "Bridging Performance and Flexibility in Network Architecture"
subtitle: "Scaleway's Experience with Spectrum-X"
taxonomy: "AI, Networking, Spectrum-X"
---
---
## Introduction: Scaleway's Focus on GPU Infrastructure

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_2.jpg)

*   **Scaleway**: A European cloud provider specializing in AI infrastructure.
*   **Services**: Offers full-service cloud solutions, including IaaS, PaaS, storage, and networking.
*   **Infrastructure**: Operates 100,000+ servers across 10+ datacenters, with over 1,000 deployed GPUs.
*   **Networking**: Utilizes InfiniBand and Spectrum-X for GPU clusters.
*   **Focus**: European sovereignty, open-source solutions, and AI infrastructure for training and inference.

---
## Understanding GPU Clusters: Essential Components and Networks

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_3.jpg)

*   **GPU Cluster Components**: GPU servers, cluster managers, switches, and fiber connectivity.
*   **Goal**: Operate multiple machines as a single, powerful system.
*   **Four Critical Networks**:
    1.  **East-West**: Inter-GPU communication (highest throughput, lowest latency).
    2.  **North-South**: Front-end connectivity to external systems (hundreds of Gbps).
    3.  **Out-of-Band (OOB)**: System management and control.
    4.  **Storage Network**: High-speed data access.
*   **Market Trends**: Growing demand for smaller clusters and integration with cloud tools.

---
## East-West Fabric: Enabling High-Performance GPU Communication

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_4.jpg)

*   **East-West (E-W) Fabric**: Specialized network for efficient GPU communication.
*   **InfiniBand Architecture**: Two-tier network with leaf and spine layers, supporting up to 2,000 GPUs.
*   **Ethernet Architecture**: Similar two-tier design, can scale to 8,000 GPUs with a "clos-3" configuration.
*   **Network Requirements**:
    *   GPU-to-GPU memory connectivity through RDMA.
    *   Highest throughput, minimal latency, and consistent jitter.
    *   Support for bursty traffic patterns.

---
## Spectrum-X: InfiniBand Performance within Ethernet

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_5.jpg)

*   **Spectrum-X**: Networking technology combining InfiniBand performance with Ethernet infrastructure.
*   **Core Features**: Enhanced Ethernet RoCE with full lossless architecture, adaptive routing, and contained tail latency.
*   **Technical Advantages**:
    *   **Flow-based routing**: Manage entire communication flows instead of individual packets.
    *   **Disordered routing with ordered delivery**: Packets take different paths but arrive in order.
*   **Why Traditional Ethernet Falls Short**: Bursty traffic, limited flow management, uneven link usage, and poor tail latency.

---
## Spectrum-X Architecture: Packet Routing in Advanced GPU Networks

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_6.jpg)

*   **Spectrum-X Network Architecture**: Multiple GPUs connected through a specialized fabric.
*   **Key Components**: GPUs, BlueField DPUs, SP-X Leaf nodes, and SP-X Spine nodes.
*   **Traffic Management**:
    *   Guaranteed Packet Delivery
    *   Global Traffic Management
    *   Dynamic Routing Adaptation
    *   Pause Mechanism
    *   Order Preservation
*   **Advantages**: Leverages Ethernet expertise, better multi-tenancy, and dynamic cluster control.

---
## Removing Pain Points: Spectrum-X Advantages over InfiniBand

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_7.jpg)

*   **InfiniBand Challenges**: Complex management, maintenance issues, limited talent pool, and customer hesitation.
*   **Spectrum-X Benefits**:
    *   Standard Ethernet technology.
    *   Extends existing infrastructure.
    *   Faster engineer onboarding.
    *   Improved troubleshooting.
    *   Cost efficiency.
*   **Implementation Challenges**: Initial deployment complexity and supply chain issues.

---
## Cluster Management: Addressing the Multi-Tenancy Challenge

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_8.jpg)

*   **Current Limitations**: Lack of a fully multi-tenant cluster manager.
*   **Essential Requirements**: Hardware awareness, VRF awareness, and centralization.
*   **Cloud Provider Challenges**: Custom integration, specialized internal tenant, tenant isolation, and storage segmentation.
*   **Scaleway's Approach**: Developed a specialized "REST tenant", implemented tenant-specific VRFs, and developed custom tooling.

---
## Spectrum-X QoS: Quality of Service for AI Workloads

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_9.jpg)

*   **QoS Technologies**: PFC, ECN, traffic prioritization.
*   **Traffic Prioritization**: 90% lossless traffic (GPU comms), 10% lossy traffic (control).
*   **Network Routing**: Designed for AI traffic patterns.
*   **Benefits**: Fairness across tenants, optimized for AI workloads, and seamless application experience.

---
## Setting Up Spectrum-X on the DPU: Configuration Steps

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_10.jpg)

*   **Configuration Steps**:
    1.  Install DOCA components.
    2.  Update networking firmware.
    3.  Enable advanced networking features.
    4.  Configure traffic prioritization.
    5.  Enable interface configurations.
*   **Benefits**:
    *   Improved performance.
    *   Operational flexibility (easy cluster slicing, tenant onboarding, reconfiguration).
    *   Automation capabilities.
*   **Implementation**: Uses VRF per tenant for multi-tenancy in an SDN environment.

---
## NVIDIA Inception Program: Supporting AI Startup Growth

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_11.jpg)

*   **Program Overview**: NVIDIA initiative to support AI, data science, and tech startups.
*   **Key Benefits**: Free developer tools, exclusive pricing, investor exposure, and additional resources.
*   **Visual Representation**: Depicts the collaborative and technical nature of the startup ecosystem.
*   **Call to Action**: Join at nvidia.com/inception.

---
## NVIDIA Connect Program for ISVs: Technical Resources for Software Vendors

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/bridging-performance-and-flexibility-in-network-architecture_output/screenshots/scene_12.jpg)

*   **Program Overview**: NVIDIA's specialized resources for ISVs.
*   **Key Benefits**: Developer resources, technical training, and preferred pricing.
*   **Visual Representation**: Green cubes depicting growth, tech elements, and optimization benefits.
*   **Call to Action**: Apply now at nvidia.com/connect-program.

---
## Detail-Oriented Takeaways

*   Scaleway, a European cloud provider, leverages Spectrum-X to enhance GPU cluster performance and flexibility.
*   Spectrum-X builds on Ethernet, offering InfiniBand-like performance for AI workloads, simplifying management, and reducing costs.
*   Key advantages of Spectrum-X include flow-based routing, dynamic traffic adaptation, and improved multi-tenancy support.
*   Implementing Spectrum-X requires careful configuration of the BlueField DPUs and integration with cloud infrastructure.
*   NVIDIA's Inception program supports AI startups with tools, resources, and funding opportunities.
*   The NVIDIA Connect program empowers ISVs to integrate NVIDIA technologies into their software, driving innovation.
---END MARP DECK---
```