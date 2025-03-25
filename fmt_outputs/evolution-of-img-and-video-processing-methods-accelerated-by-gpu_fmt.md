---
marp-theme: "proposal"
title: "Accelerating Video & Image Processing with GPUs"
subtitle: "Volcano Engine's Approach to CUDA-Accelerated Algorithms"
taxonomy: "GPU, CUDA, Video Processing, Image Processing, TensorRT"
---
## Introduction

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_1.jpg)

*   **Context:** Volcano Engine's video cloud multimedia lab utilizes GPUs to accelerate video and image processing.
*   **Focus:** Rewriting traditional algorithms using CUDA and modern CNNs with TensorRT.
*   **Goal:** Enhance and analyze video/images with improved performance and efficiency.

---
## Key Technical Concepts

*   **CUDA Acceleration:** Accelerates traditional image processing algorithms.
*   **TensorRT Optimization:** Speeds up CNN-based models.
*   **Hardware Encoding/Decoding:** Optimizes the video processing pipeline.
*   **Asynchronous Processing:** Improves system performance.

---
## Performance Improvements

*   **CUDA Boost:** Hundreds of traditional algorithms accelerated.
*   **Significant Gains:** At least a 40x performance increase.
*   **Real-world Applications:** Real-time transcoding, video enhancement, live video, and image processing.
*   **Efficiency:** An all-GPU processing chain that avoids data transfer between host and device, significantly improving computational efficiency.

---
## HDR Algorithm Acceleration with CUDA

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_2.jpg)

*   **HDR Enhancement:** Improves image vibrancy, contrast, and detail.
*   **Traditional Challenge:** Histogram-based color adjustment was slow.
*   **CUDA Solution:** Parallel processing using dynamic parallelism and Cooperative Groups.

---
## Technical Implementation

*   **Dynamic Parallelism:** GPU threads launching additional threads.
*   **Cooperative Groups:** Flexible thread synchronization (CUDA 9.0+).
*   **Performance Boost:** 200x faster than CPU on NVIDIA T4 GPUs.
*   **Practical Example:** HDR algorithm applied to a landscape photo.

---
## Enhancing Models with TensorRT

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_3.jpg)

*   **Model Optimization Pipeline:** PyTorch to TensorRT conversion.
*   **Conversion Paths:**
    1.  PyTorch -> ONNX -> TensorRT
    2.  PyTorch -> torch2trt -> TensorRT

---
## Performance Gains and Techniques

*   **Speed Increase:** Average 2.5x faster inference across 100+ models.
*   **Precision Flexibility:** Supports INT8, FP16, and TF32.
*   **Super-Resolution Example:** EDV2 model upscaling 1080p to 4K.

---
## Super-Resolution Case Study Details

*   **Initial Memory:** 50GB in FP32 (30GB in half precision), exceeding T4 memory.
*   **Model Partitioning:** Three TensorRT models, sharing workspace.
*   **Operation Fusion:** Custom CUDA kernels for efficiency.
*   **Results:**
    *   4x performance increase.
    *   Memory usage reduced from 2700MB to 800MB (400MB in half precision).

---
## GPU Utilization Challenges

*   **Three-Stage Processing:** Pre-processing (CPU), Inference (GPU), Post-processing (CPU).
*   **Synchronous Scheduling:** Can lead to GPU underutilization.
*   **Goal:** Optimize the entire pipeline for maximum performance.

---
## GPU Executor Mode for Dynamic Image Enhancement

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_4.jpg)

*   **Evolution:** From single-mode to content-adaptive enhancement.
*   **Architecture:** Parallel analysis (noise, VQScore) with algorithm selection.
*   **Process:** Condition check on CPU, processing on GPU, synchronization.

---
## Adaptive Approach Challenges

*   **Low GPU Utilization:** Below 70% (40% in adaptive mode).
*   **Excessive Memory Consumption:** Growing number of templates and models.
*   **Maintenance Complexity:** Difficulty in managing templates.

---
## GPU Executor Mode Solutions

*   **Shared Workspace:** Models share memory, reducing requirements.
*   **Resolution-Based Allocation:** Memory allocation optimized for input size.
*   **Asynchronous Scheduling:** Threads and queues to reduce CPU bottlenecks.
*   **CUDA Streams/Events:** Asynchronous GPU task submission and completion tracking.

---
## Optimization Results

*   **GPU Utilization:** Approaching 100% during peak operation.
*   **Limiting Factors:** Task distribution and CPU bottlenecks.
*   **Result:** Successfully enables content-adaptive image enhancement.

---
## GPU Executor Mode: Template Optimization

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_5.jpg)

*   **Goal:** Efficient task allocation between GPU and CPU.
*   **Task Handling:** Tasks segmented for GPU/CPU execution.
*   **Parallelism:** Multiple tasks running simultaneously.
*   **Alternating Workloads:** GPU and CPU processing blocks.

---
## Memory Optimization

*   **Block-Based Processing:** Divide images for high-resolution input.
*   **Overlap Processing:** Overlap blocks to avoid artifacts.
*   **Memory Reduction:** Consumption reduced by over 50%.

---
## GPU Utilization Maximization

*   **High Utilization:** Above 97% through multi-threading.
*   **Model Capacity:** Single A10 GPU can host 60+ models.
*   **Consolidated Service:** All business needs met in a single service.

---
## Real-World Implementation

*   **Unified Service:** Handles various business requirements.
*   **Configuration:** Supports different configurations.
*   **Techniques:** CUDA Streams, CUDA Events, and thread pools.
*   **Outcome:** Efficient resource utilization for complex pipelines.

---
## GPU Capabilities in Video and Image Processing

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_6.jpg)

*   **Benefits:**
    1.  Superior Processing Performance
        *   Accelerates traditional algorithms
        *   Supports multiple enhancement models on GPU
        *   Hardware encoding/decoding
    2.  Ease of Use
        *   CUDA environment
        *   TensorRT and TensorRT-LLM
        *   Integration with FFmpeg (Video codec SDK)

---
## Developer Advantages

*   **CUDA:** Mature and comprehensive framework.
*   **TensorRT and TensorRT-LLM:** Optimized AI model deployment.
*   **FFmpeg Integration:** Leverage hardware acceleration.

---
## Conclusion

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/evolution-of-img-and-video-processing-methods-accelerated-by-gpu_output/screenshots/scene_7.jpg)

*   **Thank You.**

---
## DETAIL-ORIENTED TAKEAWAYS

*   Volcano Engine utilizes NVIDIA GPUs and CUDA for video and image processing acceleration.
*   CUDA is used to accelerate traditional algorithms, while TensorRT is used to optimize CNN models.
*   Significant performance improvements (40x or greater) are achieved through GPU acceleration.
*   HDR algorithm acceleration demonstrates the benefits of GPU-based parallel processing with dynamic parallelism and cooperative groups.
*   TensorRT optimizes AI enhancement models, with an average speed increase of 2.5x and support for various precision formats.
*   The GPU Executor Mode addresses challenges in dynamic image enhancement through shared memory, asynchronous scheduling, and optimized task allocation.
*   The system utilizes CUDA Streams and Events for efficient GPU task management.
*   GPU utilization is maximized through multi-threading and optimized memory management, achieving over 97% utilization.
*   GPUs provide superior performance and ease of use, with the CUDA environment, TensorRT, and FFmpeg integration.
*   The presentation emphasizes the importance of GPU technology for modern video and image processing workflows.