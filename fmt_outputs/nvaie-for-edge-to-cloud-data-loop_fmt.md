---
marp-theme: proposal
title: "NVAIE for Edge-to-Cloud Data Loop: Enabling Autonomous Driving"
subtitle: "A Comprehensive Overview of WMTech and YOOCAR's Data-Driven Platform"
taxonomy: "Autonomous Driving > Data Pipeline > Closed-Loop"
---
## The Evolution of Autonomous Driving Technology

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_1.jpg)

*   Autonomous driving technology is evolving through three phases:
    *   **Phase 1: Hardware-Centric (ADAS)** – Focus on sensors (radar, LiDAR, cameras), basic data collection, and processing. Implements basic ADAS features like AEB and ACC.
    *   **Phase 2: Algorithm-Driven (L2 Automation)** – Shift to software optimization, modularization, and OTA updates.  Key components include motion control, environment sensing, decision making, and route planning. L2 automation and TJA are enabled, with data security and compliance becoming important.
    *   **Phase 3: Data-Driven (Advanced L2)** – Emphasizes closed-loop data systems, data mining, model training, and simulation. Cloud technologies enable large-scale data processing for advanced L2 systems.

---
## Data Requirements for Autonomous Driving Evolution

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_2.jpg)

*   Data requirements increase dramatically across stages:
    *   **Traditional:** Rule-based car control, specialized collection vehicles, point cloud + video + map data, in-car computing, 3D labeling, target perception, hundreds of GB/day/car, ~10 GPUs.
    *   **Mainstream:** Data-driven rules, mix of collection and mass market vehicles, point cloud + video, car + cloud computing, 4D labeling, pre-fusion + target perception, hundreds of TB/day/car, ~hundreds of GPUs.
    *   **Future/Leading:** Large AI models, mass market vehicles, video + human operation data, cloud-based processing, AI-assisted labeling, large model training, millions of cars generating GBs daily, 10,000+ GPUs.
*   The trend is toward explainable end-to-end models trained on vast amounts of data.

---
## Tesla's Industry-Leading Autonomous Driving Data Engine

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_3.jpg)

*   **Tesla's Data-Driven Approach:** Extensive data pipeline using the Autopilot data engine.
*   **Key Infrastructure:** Super-large-scale cluster (tens of thousands of H100 GPUs), $10+ billion annual investment.
*   **Data Collection:**  Shadow mode data collection from the vehicle fleet.
*   **FSD v12 Evolution:** Shift to end-to-end automated driving algorithm learning from massive data via shadow mode, resembling human-like driving.
*   **Computing Power Growth:** Projected growth to hundreds of thousands of GPU-equivalent units, ranking among the top 5 computing infrastructures globally.

---
## Challenges and Objectives of Data Closed-Loop Systems for Autonomous Driving

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_4.jpg)

*   **Current Challenges for OEMs:**
    1.  Data collection difficulties (cost, qualifications).
    2.  Data compliance issues (cost, lack of established practices).
    3.  High transmission and storage costs.
    4.  Inefficient data labeling (manual, bottleneck).
    5.  Time-consuming model training (slow data generation, insufficient computing).
    6.  Limited simulation scenarios (setup complexity, inefficient corner case building).
*   **Ideal Objectives:**
    1.  Automatic data generation (simulation tools).
    2.  Compliant data handling (data breach prevention, integrity).
    3.  Efficient data management (pre-processing, essential data upload).
    4.  AI-powered labeling (automation, efficiency).
    5.  Optimized model training (large data quantities, improved accuracy).
    6.  Distributed testing architecture (parallel processing, efficient simulations).

---
## YooDriveCloud Data Closed-loop Toolchain

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_5.jpg)

*   **YooDriveCloud Solution Overview:** End-to-end solution for autonomous driving data management.
*   **Key Features:**
    *   Closed-loop processing.
    *   Iterative development of crowd-sourced maps.
    *   Data-driven workflows (acquisition, processing, storage, training, and simulation).
*   **Data Management Components:** Data collection, processing, recharging, analysis tools, scalable framework.
*   **AIDC (AI Computing Center):** GPU clusters (NVIDIA or domestic), computing power leasing, and resource management.
*   **Toolchain Capabilities:** Annotation services (automatic labeling), distributed algorithm training, multi-task support.
*   **Model Application Services:** Model deployment marketplace, online inference, and personalized AI solutions.
*   **Key Advantages:** One-stop service, strong compatibility, and customized service.

---
## Intelligent Operation and Maintenance Management for Vehicle Networks

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_6.jpg)

*   **System Architecture:** Advanced Intelligent O&M Management Service for connected vehicle fleets.
*   **Key Components:**
    *   Service Personnel Integration: Car after-sales reps, "400 seats," Youkai experts, tripartite experts, O&M experts.
    *   Automated Processing Layers: Manual processing layer (expert access), automated/system processing layer (CMP, TSP, data platform).
    *   AI Robot Fleet: Dialogue, inspection, telepresence, document, and interface robots.
*   **Core Processing Capabilities:** Diagnostic templates, tripartite diagnosis, routine protocols, event tracking, SLA estimation.
*   **Performance Improvements:** Latency reduced from 55ms to 25ms, network jitter from 280ms to 35ms.
*   **Business Value:** Predictive maintenance, AI-powered automation, cross-manufacturer support, enhanced vehicle connectivity, dedicated vehicle network.

---
## Data Closed-Loop Compliance Services for Autonomous Driving

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_7.jpg)

*   **Architecture Overview:** Comprehensive data closed-loop compliance system.
*   **Data Flow Framework:** Three-tier cloud structure (Infrastructure, Platform, Application layers).
*   **Data Collection Pathways:** Production vehicles (GPS, mass production maps), engineering vehicles (offline routing).
*   **Compliance Cloud Processing:** Data ferry area, data usage areas (annotation, training, simulation), desensitization area.
*   **Security and Compliance Controls:** Class A qualification entity control, datastore with desensitization, VDI, special line connections, compliance workstations.
*   **Business Value:** Global connectivity services, full-stack asset product services, legal compliance, and NVIDIA partnership.

---
## Data Closed-loop Tool Chain & NVIDIA Partnership for Autonomous Driving 2.0

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_8.jpg)

*   **AV 1.0 vs. AV 2.0:**  Transition from basic data cleaning, labeling, training, and scenario replay (AV 1.0) to more advanced AV 2.0 capabilities.
*   **Comprehensive AV 2.0 Technology Stack:**
    *   **1. Data & Development Layer (AV 2.0 CICD):** Data and scene generation, Model training/customized tuning and development, auto-driving model verification.
    *   **2. Software Layer (SW NVAIE for AV CICD):**  CV CUDA, DALI, TensorRT, Triton.
    *   **3. Computing Power & Simulation Layer:** Omniverse Cloud, OVX Server, DGX Cloud, DGX SuperPOD.
*   **Enhanced AV 2.0 Capabilities:** Automated data cleansing/annotation, large model capabilities, synthetic data, scene reconstruction, world model production, scenario mining.
*   **Strategic NVIDIA Partnership:**  NVIDIA provides computing power, verification platforms, NVAIE software, support, and resources.

---
## Model Training Platform Architecture for AI and Autonomous Driving

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_9.jpg)

*   **Computing Power Base Layer:**  Kubernetes (KSE) container engine, CPU/GPU services, storage (EPFS, object, NAS), network.
*   **Computing Power Scheduling Layer:** Device plug-ins/schedulers, Computing Center (CC) management, hash pools.
*   **Computing Power Services Layer:** CC overview/management, data storage, billing, and container instances.
*   **Interface Options:** Web interface (Jupyter/VS Code), SDK/API, CLI.
*   **AI Application Support:** Autonomous driving training/inference, financial services, community/traffic management, and education/logistics.
*   **Training and Inference Capabilities:** Stand-alone/distributed training, model fine-tuning, model management, and accelerated inference.
*   **Performance Improvements:** 40% (BVFORM), 18% (UNIED), 180% (SWIN Transformer).
*   **Strategic Advantages:** Superior performance, lightweight implementation, seamless integration.

---
## Performance Optimization Through Nsight System Analysis

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_10.jpg)

*   **Analysis Methodology:** Collect system profiles with Nsight, timeline analysis, and program execution analysis.
*   **Visual Comparison of Performance:** Before and after optimization timelines showing kernel execution, memory operations, and system processes.
*   **Measurable Performance Improvement:** 27.5% reduction in forward track chain execution time (from 3.837s to 2.783s).
*   **Strategic Value:** Gain insights into system behavior and achieve targeted performance gains.

---
## BCM Cluster Management for High-Performance Computing

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_11.jpg)

*   **Accelerate the Process of Value Realization:** Streamlines deployment from bare machine to fully operational clusters.
*   **Reduce Complexity and Workload:** Enables one-click updates, prevents server drift, and offers cross-cluster job monitoring.
*   **Achieve Agility:** Supports diverse hardware environments, automatic server detection, and hybrid cloud allocation.

---
## Data Generation for Autonomous Driving

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_12.jpg)

*   **Key Challenges:** Data scarcity, extreme case collection, and labeling costs.
*   **Advantages of Synthetic Data:** Simulate complex scenarios, flexible adjustment, cost-effective.
*   **Two Key Data Generation Approaches:**
    1.  **Generalized Scene Generation:** Weather variations, road conditions, lighting.
    2.  **Multimodal Data Generation:** Camera view, segmentation maps, point cloud/LiDAR.
*   **Strategic Importance:** AV 2.0 platform supports extensive scene tags and automated scene mining.

---
## Automated Annotation with LLM Technology

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_13.jpg)

*   **Addressing Data Labeling Challenges:** Addressing labor-intensive, time-consuming, and expensive manual labeling with AI.
*   **Technical Capabilities:**
    *   Multimodal Data Processing (2D, 3D, 2D-3D fusion).
    *   Advanced Architecture Support (BEV + Transformer, YOLOR, SAM, GroundingDINO).
*   **Practical Applications:** 2D-3D fusion automatic annotation.
*   **Significance for Autonomous Vehicle Development:** Reducing bottlenecks, enabling efficient processing, supporting complex fusion, and providing consistent labeling.

---
## Rapid Iteration and Integration of Automatic Annotation Model

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_14.jpg)

*   **Data Management Platform Components:** Data set, operator warehouse, data processing model, and computing power center.
*   **Key Technical Capabilities:**
    1.  Industry Large Models for Automatic Annotation.
    2.  Customization Flexibility.
    3.  Monthly Iteration Cycle.
    4.  Full-stack Technical Capability.
    5.  Stable Platform Architecture.

---
## Automatic Annotation-Driven Data Generation and Simulation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_15.jpg)

*   **1. Data Collection:** Sensor hardware, 3D point clouds, road imagery, and vehicle models.
*   **2. Automatic Labeling Process:** Data analysis/cleaning, 3D point cloud scene modeling, automatic labeling with manual quality inspection, static/dynamic background reconstruction.
*   **3. Data Generation and Simulation:** 3D scene reconstruction, vehicle dynamics, traffic flow models, sensor model libraries, and test scenario generation.
*   **Key Technical Innovation:** Data generation and validation, creating static backgrounds, dynamic trajectories.

---
## Self-Driving Algorithm Development Platform for OEMs

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_16.jpg)

*   **Closed-Loop Development Solution:** Integrated platform for algorithm development and testing for OEMs.
*   **Platform Architecture and Components:** Data handling, model development, simulation, and deployment.
*   **Business Pain Points Addressed:**
    1.  Disjointed Development Process.
    2.  Low R&D Efficiency.
*   **Solution Benefits:**
    1.  Process Integration.
    2.  R&D Efficiency Improvements.
*   **Strategic Vision:** Provides OEMs with an integrated vehicle-cloud intelligence solution.

---
## Conclusion of the WMTECH and YOOCAR Technical Presentation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_17.jpg)

*   **Key Takeaways:**
    1.  Closed-Loop Development System:  Connects annotation, training, and simulation.
    2.  Efficiency Improvements: Addresses pain points, reduces iteration cycles.
    3.  Integration Capabilities: Creates a cohesive workflow.
*   **Strategic Vision:**  Empowering global automakers to lead in AI transportation.

---
## Closing Frame of the Technical Presentation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/nvaie-for-edge-to-cloud-data-loop_output/screenshots/scene_18.jpg)

---
## DETAIL-ORIENTED TAKEAWAYS

*   Autonomous driving is evolving from hardware-focused to data-driven approaches.
*   Data requirements for autonomous driving are increasing exponentially, with cloud computing playing a vital role.
*   Tesla's data engine is a benchmark, utilizing large-scale data for end-to-end AI model training.
*   WMTech and YOOCAR offer a comprehensive data closed-loop platform, addressing challenges in data collection, compliance, and model training.
*   The platform includes intelligent O&M services for vehicle networks with significant performance improvements.
*   Data compliance is ensured through secure data handling and controlled access.
*   The platform leverages NVIDIA's NVAIE and other key technologies for AV 2.0 capabilities.
*   The platform incorporates automated annotation using LLMs.
*   The platform provides a self-driving algorithm development platform for OEMs, solving key industry pain points.
*   WMTech and YOOCAR are committed to empowering global automakers to lead in the AI era, offering a comprehensive solution for autonomous driving development.