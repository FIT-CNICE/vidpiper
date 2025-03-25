---
marp-theme: proposal
title: "A Deep Dive into Neural Networks for Radar Perception"
subtitle: "Unlocking Advanced Sensing Capabilities"
taxonomy: "Perception > Radar > Neural Networks"
---
## Radar-only Perception: Unlocking Advanced Sensing Capabilities

![width:500px](screenshots/scene_2.jpg)

- **Radar-only perception** is a focus, showing object detection and road area identification.
- **Camera-only perception** is shown for comparison.

---
## Key Technical Concepts

- AI research focused on computer vision and NLP, but radar offers key advantages.
  - Works in darkness, fog, and rain.
  - Directly measures speed (Doppler velocity).
  - Lower cost than LiDAR.

---
## The Challenge and AI Solution

- Traditional radar limitations:
  - Complex data processing
  - Handcrafted features (CFAR, clustering, Kalman filtering)
  - Higher noise levels

- AI's potential:
  - Data-driven approaches outperform handcrafted features.
  - Reimagining radar as a comprehensive perception system.

---
## Scene 3 - Neural Networks & Radar: A Natural Fit for Advanced Sensing

![width:500px](screenshots/scene_3.jpg)

---
## Why Radar and AI Work Well Together

1.  **High-dimensional data**: Radar combines velocity, range, and angle data.
2.  **Pattern extraction**: AI finds info within noisy data.
3.  **Adaptability**: AI learns and adapts to environments.

---
## Provizio's Innovation Approach

- They build radar sensors with hardware & software innovation:
  - Proprietary RFICs
  - Custom imaging radar systems
  - Advanced demultiplexing
  - Super-resolution algorithms

- Result: longer range, better angular resolution, and higher density point clouds.

---
## Balancing Hardware Constraints with AI Capabilities

- Tradeoffs in L3+ advanced driving assistance:
  - Cost reduction = fewer radar chips
  - Fewer elements = lower angular resolution
  - Lower resolution = less detailed point clouds

- AI's value: Extracting meaningful info from less detailed data.

---
## The Labeling Challenge

- Illustrating radar's complexity:
  - Camera: Easy for humans to label.
  - Radar: Challenging to identify objects.

- AI's role: Neural networks embed human reasoning and extract patterns.

---
## Scene 4 - Neural Network Architectures Designed for Radar Data

![width:500px](screenshots/scene_4.jpg)

---
## Data Processing Approaches

1.  **Raw Radar Data Processing**:
    - Processes signals early in the chain.
    - Preserves fine-grained features.
    - AI learns from raw data.
    - Challenge: "Black box" and difficult labeling.
2.  **Point Cloud Processing**:
    - Uses AI on pre-processed data.
    - Benefits from existing research.
    - Easier to implement.
    - Drawback: Filtering might remove crucial details.

---
## Unique Characteristics of Radar Point Clouds

-   **Doppler features**: Unique velocity measurements.
-   **Higher noise levels**: More background noise.
-   **Different sparsity patterns**: Unique distribution characteristics.

---
## Neural Network Architecture Components

### Encoding Methods:
- Pillar-based
- Range projections
- Voxel-based

### Backbone Structure:
- Spatiotemporal feature extraction
- Multi-scale architectures
- Attention mechanisms (local and temporal)

### Perception Heads:
- Object detection
- Occupancy/Freespace
- Motion estimation

---
## Scene 5 - Radar-only Freespace Estimation: Combining SLAM and Bayesian Techniques

![width:500px](screenshots/scene_5.jpg)

---
## The Challenge of Freespace Estimation

- Identifying areas for safe navigation.
- Challenges:
  - Complexity in defining "freespace."
  - Sparse radar data and lack of semantic info.

---
## Proposed Solution Approach

- Combining:
  - **SLAM (Simultaneous Localization and Mapping)**: Maps and tracks location.
  - **Bayesian algorithms**: Updates predictions with new evidence.

- Generates a "supervision signal on-the-fly."

---
## Connection to Earlier Topics

- Related to multitask learning.
- Addresses limited computation resources.
- Addresses challenges in radar data labeling.

---
## Key Takeaway

- Radar-only freespace estimation is challenging.
- Combining mapping (SLAM) and probabilistic methods (Bayesian) enables training neural networks.
- It avoids extensive manual labeling of training data.

---
## DETAIL-ORIENTED Takeaways

-   Radar-only perception is a viable alternative to camera-based systems, offering advantages in challenging conditions.
-   AI is crucial for unlocking radar's potential due to its ability to handle complex data and noisy signals.
-   Neural networks are well-suited for radar data processing due to its high dimensionality, noise, and adaptability.
-   Provizio's approach involves both hardware and software innovations to enhance radar sensing capabilities.
-   Neural network architectures must consider radar's unique characteristics to extract meaningful features effectively.
-   Freespace estimation using radar data is challenging but can be addressed by combining SLAM and Bayesian algorithms for training.