---
marp-theme: proposal
title: "Unlocking Biological Discovery with AI: BioMap's Approach"
subtitle: "Accelerating Life Science Innovation with AI Foundation Models"
taxonomy: "AI > Life Sciences > BioMap"
---
## The Challenge of Protein Design in Biological Discovery

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_1.jpg)

*   **Problem**: Traditional methods struggle to explore the vast landscape of biological possibilities.
*   **Opportunity**: Huge potential in areas like precision medicine and novel therapies.
*   **Visual**: Contrasts the tiny "known space" with the immense "unknown space" of potential discoveries.
*   **Core Challenge**: Efficiently finding valuable biological innovations within an enormous search space.

---
## xTrimo: A Comprehensive Foundation Model for Multi-Modal Biological Data

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_2.jpg)

*   **Introducing xTrimo**: A 210 billion parameter AI foundation model.
*   **Key Modules**:
    *   xTrimoDNA: Decodes genomic information.
    *   xTrimoRNA: Models RNA functions.
    *   xTrimoProtein: Analyzes and designs proteins (100B parameters).
    *   xTrimoCell: Models cellular targets (first of its kind).
    *   xTrimoChem: Supports drug design.
    *   xTrimoPPI: Models protein interactions.
    *   xTrimoSystem: Analyzes therapy mechanisms.
*   **AI Approach**: Uses biological "language" and massive datasets for better generalization.

---
## Life Science Foundation Models Differ Fundamentally from NLP Models

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_3.jpg)

*   **Beyond NLP**: Life science AI models are distinct.
*   **Key Differences**:
    *   **Computing Power**: Similar GPU cluster needs.
    *   **Validation**: Wet-lab validation is essential, BioMap has in-house facilities.
    *   **Model Architecture**: Custom bio-networks target biological mechanisms.
    *   **Data Foundation**: Uses bio-omics data like genomics and proteomics.
    *   **Application Layer**: Target discovery and drug design vs. medical consultation.
*   **BioMap's Advantage**: World's largest bio-computing cluster.

---
## GPU Optimization Strategies for Large Protein Models

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_4.jpg)

*   **Optimization Techniques**:
    *   **4D Hybrid Parallelism**: Optimizes communication between experts (1.5x performance).
    *   **FP8 Quantization**: Uses 8-bit floating point (1.25x speedup) with negligible accuracy loss.
    *   **Smart Activation Recompute**: Intelligent layer recomputation (1.2x acceleration).
    *   **Elastic Training Service**: Ensures >99% effective training utilization.
*   **Genomic Modeling**: Addresses challenges of extremely long genomic sequences.

---
## RNAGenesis: Advanced RNA Foundation Model Architecture

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_5.jpg)

*   **RNAGenesis**: A dedicated RNA foundation model.
*   **Model Architecture**:
    *   Specialized Encoder: Hybrid n-gram tokenization and CNN layers.
    *   Innovative Decoder: Classifier-guided diffusion for RNA design.
*   **Performance**: Achieves state-of-the-art results, especially for aptamer design.
*   **Biological Considerations**: Incorporates RNA-specific principles and secondary structure modeling.

---
## OminiBio: Cross-Modal Training Framework for Life Sciences

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_6.jpg)

*   **Introducing OminiBio**: A cross-modal training framework.
*   **Framework Architecture**: Built on Megatron, including:
    *   Foundation Layers: Core transformer architecture.
    *   Biological Cross-modal Foundation: Supports multiple modalities.
    *   Pre-training and Fine-tuning Integration: PEFT and biological task fine-tuning.
*   **Key Innovations**:
    *   First cross-modal framework for life sciences.
    *   Unified training pipeline.
    *   Dedicated I/O optimization.

---
## Accelerating Mini-Protein Design Through Optimized Structure Prediction

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_7.jpg)

*   **Challenge**: Mini-protein design is computationally expensive.
*   **Solution**: xTrimoPGLM, a GPU-accelerated solution.
*   **Traditional Bottlenecks**: MSA and template searching on CPUs.
*   **High-Performance Solution**: xTrimoPGLM processes amino acid sequences and runs entirely on GPUs.
*   **Performance Results**: Achieves comparable or better accuracy and faster inference (50 seconds).

---
## Accelerating MSA Tools for Faster Protein Structure Prediction

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_8.jpg)

*   **Focus**: Optimizing Multiple Sequence Alignment (MSA).
*   **Performance Improvements**:
    *   JackHMMER Optimization: 8.6x speedup.
    *   MMseqs2 GPU Acceleration: 186.7x speedup.
*   **Impact**: Faster MSA, reducing bottlenecks in structure prediction.

---
## MSA Generator Optimization for Improved Protein Structure Prediction

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_9.jpg)

*   **Challenge**: Limited natural MSAs.
*   **Solution**: Synthetic MSA generation.
*   **Technology Pipeline**: Amino acid sequence -> xTrimoPGLM -> MSA Generator -> xTrimoPGLMFold.
*   **Performance Benefits**: Improved accuracy.
*   **NVIDIA Collaboration**: Smooth Quantization, Flash Decoding and other optimization techniques for speed.

---
## Biology-AI Fusion: An Integrated Distributed Inference Engine for Life Sciences

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_10.jpg)

*   **Challenge**: Fusing traditional HPC and AI inference.
*   **Solution**: A Biology-AI Fusion framework.
*   **Framework**: Built on Ray and TensorRT-LLM.
*   **Dramatic Improvements**: Over 10x performance boost in protein design time.
*   **Workflow Example**: AI-driven protein design pipeline for enhanced efficiency.
*   **Key Components**: Bio-AI Node Library, Graph Builder, Ray Core, TensorRT-LLM.

---
## Generative Discovery System: Accelerating Life Science Innovation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_11.jpg)

*   **Introducing the System**: Comprehensive Generative Discovery System.
*   **Three-Stage Workflow**:
    1.  Knowledge Assistant: Literature mining and project selection.
    2.  Rational Design: AI-driven optimization and generative models.
    3.  Intelligent Experiments: Automated high-throughput validation.
*   **Key Feature**: Continuous improvement through automatic iteration.
*   **Foundation**: Foundation Model, Data, Multi-Task Models, High-throughput System.
*   **Impact**: Significantly accelerates discovery processes.

---
## Life Science Discovery System: A Multi-Agent Intelligence Architecture

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_12.jpg)

*   **Multi-Agent Architecture**: Accelerating discovery with intelligent agents.
*   **System Overview**: Includes user interaction, perception, brain, and action systems.
*   **Key Components**:
    *   Knowledge Assistant Agent
    *   Rational Design Agent
    *   Intelligent Experiment Agent
*   **Technical Highlights**:
    *   Knowledge graphs based on biological principles
    *   GraphRAG
    *   Feedback loops with experimental results

---
## Global Reach of BioMap's AI Foundation Models

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_13.jpg)

*   **Global Presence**: BioMap's international locations.
*   **Locations**: Silicon Valley, Beijing, Suzhou, and Hong Kong.
*   **Goal**: Positioning AI foundation models globally.

---
## Conclusion

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/gpu-accelerated-intel-in-life-science_output/screenshots/scene_14.jpg)

---
## Key Takeaways

*   **Challenge**: Biological discovery is limited by the vastness of the design space.
*   **Solution**: BioMap has built a comprehensive AI foundation model.
*   **Advantage**: BioMap's AI models are designed for specialized biological data.
*   **Innovation**:
    *   Specialized models like xTrimo, RNAGenesis, and OminiBio.
    *   Optimization strategies to accelerate the use of GPU resources.
    *   Multi-agent system to achieve continuous improvement.
*   **Impact**: Dramatically accelerates the life science discovery process.
*   **Vision**: A global presence dedicated to accelerating scientific progress.