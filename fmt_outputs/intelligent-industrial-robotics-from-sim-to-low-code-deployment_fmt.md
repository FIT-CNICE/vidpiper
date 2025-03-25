---
marp-theme: proposal
title: "Intelligent Industrial Robotics: From Simulation to Low-Code Deployment"
subtitle: "Enabling Cognitive and Adaptive Automation in Manufacturing"
taxonomy: "Robotics > Automation > AI-Driven"
---
## Introduction: The Future of Manufacturing
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_1.jpg)
### Addressing Key Industry Challenges
*   **Decreasing Workforce:** Aging populations leading to labor shortages.
*   **Changing Production:** Shift towards customized products and smaller batch sizes.
*   **Skills Gap:** Lack of qualified workers for complex processes.
*   **Emerging Markets:** Automation needed in previously non-automated sectors.
### Solution: Cognitive and Adaptive Automation
*   Intelligent systems that adapt to changing needs.
*   Intuitive interfaces for human operators.
---
## MOTOMAN NEXT: Advanced Robot Controller Architecture
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_2.jpg)
### Enhancing Traditional Robots with AI
*   **RCU (Robot Controller Unit):** Handles standard robot functions (motion, variables). Runs on VxWorks.
*   **ACU (Autonomous Control Unit):** Powered by NVIDIA Jetson Orin™. Runs WindRiver Linux. Enables advanced AI & computing. Hosts containerized services.
### Key Services and Capabilities
*   Robot Control Service, Path Planning, Machine Vision, Force Control, AI Service, ROS2 Application.
### Technical Architecture Benefits
*   User apps run as Docker containers.
*   APIs for easier integration.
*   Open development platform.
*   Custom apps on the edge.
*   RPC communication between RCU and ACU.
---
## Smart Human-Robot Collaboration
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_3.jpg)
### Process Flow: Combining Synthetic Data and Computer Vision
1.  **Object Detection with Synthetic Data:**
    *   Isaac Sim & YOLOv8 for synthetic training data.
    *   Leverages existing CAD models.
    *   Simulated lab environment.
2.  **Human Gesture Recognition:**
    *   Google's MediaPipe for gesture identification.
    *   Ensures safe handovers.
3.  **3D Environment Simulation:**
    *   Real-time 3D simulation in Isaac Sim.
    *   Visualizes robot movements.
    *   Connected through ROS.
4.  **Hand Position Recognition:**
    *   Detects 3D hand position using MediaPipe.
    *   Enables accurate part delivery.
5.  **Task Execution:**
    *   Robot performs handover.
### Skill-Based Architecture: Modular Robot Programming
*   **Motion Primitives:** Basic robot actions.
*   **Skills Hierarchy:** Aggregation and encapsulation.
*   **Reusability:** Combining lower-level skills.
---
## Skill Implementation Architecture
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_4.jpg)
### Core Components of a Skill
1.  **FeasibilityCheck:** Task possibility validation.
2.  **PreconditionCheck:** Execution condition verification.
3.  **Execution:** Skill performance.
4.  **Monitoring:** Runtime information gathering.
5.  **ParameterSet:** Configuration settings.
6.  **FinalResultData:** Output and metrics.
7.  **StateMachine:** Execution state tracking.
### State Machine Diagram
*   Halted, Ready, Running, Completed, Suspended.
### Practical Implementation
*   **FeasibilityCheck:**  Task viability and time estimation.
*   **PreconditionCheck:** Safety checks.
*   **Execution Components:**
    *   Monitoring info (distance, etc.).
    *   Parameter settings.
    *   Final results (time, energy).
---
## Parametrized Skills for Robot Task Execution
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_5.jpg)
### "Give Skill" Structure
*   Give Skill: `get_Hand_Pos`, `get_Object_Pos`.
*   Pick Skill: Object retrieval (position, name).
*   Place Skill: Object placement (position, name).
### Advantages of the Skills Approach
1.  **Natural Language Integration:**  Ease of orchestration.
2.  **Error Recovery:** State tracking prevents errors.
3.  **Operator Accessibility:** Intuitive interface.
4.  **Modular Orchestration:** Flexible production planning.
### Natural Language Control
*   Speech-to-text system (Google Cloud, then edge).
*   BERT model for command and parameter recognition.
*   Action and parameter extraction.
---
## LLM and Behavior Tree Integration
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_6.jpg)
### Behavior Tree Architecture
*   **Command Processing Pipeline:** Speech-to-text -> BERT -> Behavior tree.
*   **Behavior Tree Structure:**
    *   Tasks/BEHAVIOR nodes.
    *   Skills as nodes (e.g., "pick," "place").
    *   Execution flow (top-to-bottom, left-to-right).
    *   Example:  "servoON -> pick -> place -> servoOFF".
### Advantages of Behavior Trees
*   Modular composition, hierarchical representation, adaptability, AI integration, natural language creation, efficient development, error handling.
### Implementation Details
*   Docker container on ACU.
*   Libraries, services, APIs.
*   BERT model and tokenizer.
*   Behavior tree creation.
*   Communication with RCU via gRPC.
---
## [Transitional Scene - Loading Animation]
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_7.jpg)
### Brief transition
---
## Robot Command Demonstration
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_8.jpg)
### Natural Language Command System in Action
1.  **Command Interface Log:** Shows LLMsBT system logs.
2.  **Robot Simulator:** 3D simulation environment.
3.  **Command Processing Flow:** Speech-to-text -> LLM -> Robot instructions -> Behavior trees.
### Command Sequence
1.  "pick the wheel" -> PickObject Behavior.
2.  "lift the chassis" -> Rejected (already holding).
3.  "put down the wheel" -> PlaceObject command.
---
## Simulation-to-Reality Implementation
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_9.jpg)
### Digital Twins and Skill Transfer
1.  **Digital Twin Synchronization:**  Virtual and real workspace.
2.  **Computer Vision Integration:** Object recognition.
3.  **Skill Transfer Pipeline:**  Simulation to real robot.
### Demonstration
*   Skill developed in simulation.
*   Camera angles match physical setup.
*   Motion executes simultaneously.
*   Controlled via teach pendant.
---
## Human-Robot Handover Task
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_10.jpg)
### Practical Application of Collaboration
1.  **Robotic Workcell:** Cobot, conveyor, safety enclosure.
2.  **Human-Robot Interaction:** Handover task.
3.  **Monitoring Systems:** Skeletal tracking.
---
## Human-Robot Collaboration with Pose Estimation
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_11.jpg)
### Leveraging Advanced Vision and Pose
1.  **Skeletal Tracking:** Digital "skeleton" representation.
2.  **Multiple Perspectives:**  Factory floor and close-up views.
3.  **Gesture-Based Control:** Gesture to initiate operation.
---
## Hand Tracking for Precise Robot Control
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_12.jpg)
### Precision in Industrial Settings
1.  **Hand Skeleton Detection:**  3D hand position tracking.
2.  **Real-Time Robot Response:**  Robot adjusts to hand position.
3.  **Industrial Implementation:**  Factory/lab environment.
---
## Conclusions and Future Developments
![width:500px](/home/fit-sizhe/dwhelper/gtc/sum_outputs/intelligent-industrial-robotics-from-sim-to-low-code-deployment_output/screenshots/scene_13.jpg)
### Key Technological Achievements:
*   NVIDIA Isaac Sim for virtual training.
*   Human posture and hand position tracking.
*   Plant orchestration system.
*   Skills-based architecture using OPC UA.
*   Natural Language and Behavior Tree Orchestration.
### Areas for future development
*   Expanding gestures in Isaac Sim.
*   Migrating speech-to-text to ACU.
*   Improving error handling.
---
## Detail-Oriented Takeaways

*   The presentation addresses industry challenges such as workforce reduction, changing production requirements, skill gaps, and the need for automation in emerging markets.
*   Yaskawa's MOTOMAN NEXT robot generation features a two-part controller architecture with a Robot Controller Unit (RCU) for standard functions and an Autonomous Control Unit (ACU) powered by NVIDIA Jetson Orin for advanced AI capabilities.
*   The system uses synthetic data generated in Isaac Sim and the YOLOv8 model for object detection, alongside MediaPipe for human gesture and hand position recognition, to enable human-robot collaboration in part handover tasks.
*   A skills-based architecture is implemented, structuring robot capabilities with core components, including FeasibilityCheck, Execution, Monitoring, ParameterSet, and StateMachine with modular functions.
*   The presentation describes how LLMs and behavior trees are integrated to create a flexible command system where speech-to-text conversion and a fine-tuned BERT model translate commands into actionable robot behaviors, which are then visualized, debugged, and modified.
*   A practical implementation is demonstrated using digital twins and simulation-to-reality transfer, showing how skills developed in simulation can be executed on physical robots and using vision systems.
*   The presentation describes the use of human posture and hand position tracking systems for gesture recognition to enable intuitive human-robot collaboration.
*   The demonstration shows the use of hand position tracking for precise placement tasks in an industrial workcell, allowing workers to indicate object placement by moving their hands, improving work efficiency.
*   The presentation concludes with the acknowledgment of ongoing development areas, including expanded gestures, speech-to-text migration, and error-handling improvements.