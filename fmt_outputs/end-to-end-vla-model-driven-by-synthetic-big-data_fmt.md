---
marp-theme: proposal
title: "End-to-End VLA Model Driven by Synthetic Big Data"
subtitle: "Enabling Generalist Robots with Efficient Learning"
taxonomy: "Robotics > VLA > Synthetic Data"
---
## Introduction: Generalist Robots and VLA Models

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_1.jpg)

*   **The Vision:** Generalist robots, capable of diverse tasks, are the future of robotics.
*   **Key Concept: VLA Models:** Vision-Language-Action models integrate vision, language understanding, and action generation.
*   **The Model Components**:
    *   Language input: Commands like "pick up the bottle"
    *   Visual Data: Information from cameras
    *   Action Output: Commands to move the robot's joints
*   **The Hardware**: Mobile base, robotic arm, gripper.
*   **The Benefit:** Easier interaction using natural language and adaptable learning with zero-shot or few-shot learning.
---
## Disembodied vs. Embodied AI: A Classification

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_2.jpg)

*   **Two Foundational AI Model Types:**
    *   **Disembodied Models:** Operate in the digital world.
        *   LLMs (e.g., GPT-4): Process and generate text.
        *   VLMs (e.g., GPT-4O): Handle text and images.
        *   Video Generation Models (e.g., SORA): Create videos from text/images.
    *   **Embodied VLA Models:** Interact with the physical world.
        *   Autonomous Driving Systems (e.g., Tesla FSD): Navigate using visual input and steering/acceleration commands.
        *   Generalist Robots (e.g., Google RT-2, GraspVLA): Process vision/text to produce physical actions.
*   **Key Differences:** Input and action complexity varies significantly.
*   **The Challenge:** Embodied models require bridging the understanding of language/vision and physical action.
---
## The Data Bottleneck: Data Collection Challenges

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_3.jpg)

*   **The Problem:** Training VLA models is data-intensive, with real-world demonstrations often required.
*   **Traditional Approaches:**
    *   **Teleoperation:** Human operators guide robots to collect training data.
        *   Examples: Tesla's and Stanford's ALOHA systems.
*   **Resource Intensity:**
    *   Google's RT-1 dataset involved 130,000 demonstrations, 16 human operators, 13 robots, and 17 months.
    *   Contrast with autonomous driving data from customer vehicles.
*   **Industry Response:** Outsourcing and collaborative data collection to address the limitations.
---
## Synthetic Data: A Scalable Solution

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_4.jpg)

*   **Limitation of Existing Approaches**
    *   Single-robot specialized data: Expensive and less effective with hardware changes.
    *   Multi-robot heterogeneous data: Accepting differences across various robots can be suboptimal for training high-performance systems for a specific robot.
*   **The Solution: Synthetic Data:** Generate artificial training data in simulated environments and transfer it to the real world (Sim2Real).
*   **Key Components:**
    *   **Object Assets (GAPartNet):** 3D object models with part information.
    *   **Action Data (DexGraspNet):** Million-scale dexterous grasping dataset.
    *   **Depth Sensor Simulation:** Simulating depth sensors to understand sensor artifacts and failure cases.
*   **The Goal:** Train powerful VLA models without any real-world data through Sim2Real.
---
## NaVid: Navigation with Synthetic Training

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_5.jpg)

*   **NaVid: End-to-End Navigation Model:**
    *   Processes vision and language inputs to navigate.
    *   Understands instructions like "walk out of the bedroom, turn right."
    *   Functions as a single end-to-end model.
*   **Training Data:**
    *   Based on the R2R dataset using the VLN-CE simulator.
    *   10,819 episodes across 61 scenes.
*   **Data Types:**
    *   Action Planning Samples: Video segments paired with robot actions.
    *   Instruction Reasoning Samples: Instructions with navigation.
    *   General VQA Data: Additional visual question-answering.
*   **Approach:** Combines embodied (navigation-specific) and disembodied (general visual understanding) data.
*   **Benefit:** Testing in unseen environments without further real-world training.
---
## Generalizable End-to-End Vision-Language Navigation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_6.jpg)

*   **Demonstration: Complex Instruction Navigation**
    *   Robot follows instructions: "Go straight and move close to the plant, then turn right facing the door, then walk to the door and stop."
*   **Key Capabilities:**
    *   Environment Generalization: Navigation in unseen environments.
    *   Single End-to-End Model: No mapping, odometry, or object detection modules needed.
    *   Enhanced Performance: Better than multi-modal models like GPT-4V/4.0 for physical navigation.
*   **Future Work:**
    *   Enhancing Spatial Intelligence using depth information.
    *   New research for object search and dynamic following tasks.
---
## Enhanced Spatial Intelligence: NaVid-4D

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_7.jpg)

*   **NaVid-4D: Depth Perception:**
    *   Uses RGB images (color) and depth information.
    *   Depth map shows distance.
*   **Benefits of Depth:**
    *   More accurate size comparisons.
    *   Understanding relative distances (closer vs. farther).
    *   Following complex spatial instructions.
*   **Technical Innovation:** The first RGBD end-to-end VLN model.
*   **Further Research:** Handle more natural instructions, like object search and following.
---
## Emergent Intelligence: Knowledge Transfer in Navigation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_8.jpg)

*   **Multi-Task Capabilities:**
    *   Single system capable of multiple tasks.
    *   Robot dog navigating and following.
*   **Training Dataset:** 3.6 million trajectories across different tasks.
*   **Instruction Examples:** Combining multiple abilities, like directions, object identification, and following.
*   **Emergent Behavior:** The system developed the ability to follow a robot dog, though not trained on this.
*   **Technical Significance:**
    *   Cross-domain knowledge sharing.
    *   Enhanced spatial intelligence.
    *   Reduced marginal learning cost.
---
## Synthetic Data for Advanced Manipulation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_9.jpg)

*   **Synthetic Data Pipeline:**
    1.  Input: Robot models, object assets, material assets.
    2.  Processing Engine: Galbot's synthetic pipeline, NVIDIA rendering.
    3.  Output: Billion frames of synthetic data.
*   **Scale and Diversity:** Billion frames of robotic grasping data.
    *   Variations in environments, objects, lighting, grasps, and instructions.
*   **Generalization Capabilities:**
    *   Visual: illumination, background, 3D position generalization.
    *   Language: category-level, open vocabulary grasping.
    *   Performance: real-time closed-loop reaction, dynamic handling.
---
## Generalization Across Diverse Heights: Robust Manipulation

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_10.jpg)

*   **Demonstration:** Robotic arm manipulating objects at different heights and positions.
*   **Capabilities Demonstrated:**
    *   Illumination Robustness: Functions in varied lighting conditions.
    *   Real-time Adaptive Control: Adjusts trajectory mid-operation.
    *   Background Generalization: Works reliably in different backgrounds.
    *   Height and Position Adaptability: Manipulates objects at various positions.
    *   Open Vocabulary Recognition: Identifies and grasps novel objects.
*   **Significance:** Model trained only on synthetic data.
---
## Few-Shot Post-Training: Learning Human Preferences

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_11.jpg)

*   **Few-Shot Post-Training:** Training robots with minimal real-world data.
*   **Requirement:** Only 200 trajectories, collected in 4 hours by one person.
*   **Purpose:** Teaching specific human preferences.
*   **How It Works:**
    1.  Pre-trained model can grasp objects, but doesn't know preferences.
    2.  By showing the robot grasping examples, it learns preferences.
*   **Comparison:** Similar to RLHF (Reinforcement Learning from Human Feedback) in language models.
*   **Generalization:** Applying learned patterns to various objects.
---
## GraspVLA: A New Paradigm for Robotic Learning

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_12.jpg)

*   **GraspVLA: Key Principles**
    1.  **Pre-training**: high-quality, large-scale synthetic data for universal grasping.
    2.  **Post-training**: minimal real data needed for alignment.
    3.  **Scalable data solution**: avoids data burden and enables efficient deployment.
*   **Demonstration of Generalization:** handling new bottles after pre-training.
*   **New VLA Paradigm**
    1.  Pre-training on synthetic data for one robot type.
    2.  Post-training with real-world data, if needed.
*   **Applications:** Retail and reception scenarios.
---
## Real-World Applications: ZEEKR & Healthcare

![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/end-to-end-vla-model-driven-by-synthetic-big-data_output/screenshots/scene_13.jpg)

*   **ZEEKR Car Factory:**
    *   Mobile robot with an articulated arm.
    *   Vision-guided operations to identify and handle boxes.
    *   Adaptive handling strategies, accessing different shelf levels.
*   **Additional Applications:**
    *   Healthcare/Pharmacy Automation: Order fulfillment, restocking.
    *   Mercedes-Benz Factory: Identifying and correcting rooftop errors.
*   **Technical Significance:**
    *   Recognizes and handles different objects.
    *   Applies handling strategies.
    *   Navigates complex environments and maintains performance.
---
## Takeaways

*   **VLA Models are Key:** Vision-Language-Action models are crucial for generalist robots.
*   **Synthetic Data Solves Bottlenecks:** Synthetic data reduces the reliance on large real-world datasets for training.
*   **NaVid Enables Navigation:** Using a single end-to-end model allows for generalization in new environments.
*   **Depth Perception Enhances Spatial Intelligence:** Improves a robot's spatial understanding.
*   **Few-Shot Post-Training Enables Efficient Learning:** Robots can learn human preferences with very little real-world data.
*   **GraspVLA Provides a New Paradigm:** A comprehensive approach to building robots.
*   **Real-World Applications are Growing:** ZEEKR and healthcare demonstrate the practical applications of this technology.

---END MARP DECK---