---
marp-theme: proposal
title: "Accelerating Automotive Design with AI and NVIDIA Omniverse"
subtitle: A Summary for a General Audience
taxonomy: "Automotive Design > AI > Omniverse"
---
## Introduction: Transforming Automotive Design

![width:500px](screenshots/scene_1.jpg)

*   **NVIDIA Omniverse for Creative Workflows:** This presentation explores how NVIDIA's Omniverse platform and AI tools are revolutionizing automotive design.
*   **Real-Time Adjustments:** Creative adjustments become more flexible and happen in real-time, improving productivity.
*   **Key Technologies:** Includes Omniverse Composer, AI image generation, and tools like KonfaUI, Laura, and Meta Journey.
*   **Goal:** Designers create more diverse and richer designs in less time through AI assistance.

---
## Unified 3D Asset Management: The Foundation

![width:500px](screenshots/scene_2.jpg)

*   **Challenge:** Managing diverse 3D assets from different software platforms in automotive design.
*   **Problem:** Various file types, formats, and workflows create inefficiencies.
*   **Solution (Part 1: Nuclear as a Core Asset Library):**
    *   Secure storage and version control of 3D assets.
    *   Track design changes efficiently.
*   **Solution (Part 2: OpenUSD as a Unified Format):**
    *   Converts data from different software into a single USD format.
    *   Enables smooth data flow between teams.
    *   Allows all team members to access and collaborate on the same assets.

---
## Knowledge Sharing with RAG Technology

![width:500px](screenshots/scene_3.jpg)

*   **RAG Technology Implementation:**  Retrieval-augmented generation (RAG) helps share 20 years of knowledge.
*   **RAG Web Service (IATChat):** Allows employees to ask questions and receive detailed responses about design aesthetics.
*   **RAG Omniverse Service:** Integrates knowledge access directly into the Omniverse platform.
*   **Benefits:**
    *   Seamless knowledge transfer, especially for new employees.
    *   Microservices architecture for flexible integration.
    *   Cross-departmental collaboration.

---
## AI Applications: Revolutionizing Design

![width:500px](screenshots/scene_4.jpg)

*   **AI-Powered Design Visualization:** Transforms sketches into fully rendered designs using GPU acceleration.
    *   Real-time adjustments to interior details.
    *   Manipulation through text prompts for materials, lighting, and styles.
    *   Generation of different design variations.
*   **AI in CAD and CFD:**  Improves efficiency and drives innovation.
    *   AI in CAD: Intelligent component generation.
    *   AI in CFD: Performance prediction using machine learning.

---
## Data Vectorization: The AI's Language

![width:500px](screenshots/scene_5.jpg)

*   **Data Transformation for AI:**
    *   3D CAD Data: Processed using PointNet to convert CAD models into numerical representations.
    *   Design Requirements: Processed using BERT to convert text specifications into numerical representations.
*   **Data Vectorization:**
    *   Converting complex information into fixed-length vectors (numerical arrays).
    *   Creates a common format for both text and 3D data.
    *   Enables the AI to find patterns and relationships.
*   **Significance:** Enables multi-modal alignment.  AI understands links between text and 3D models, enabling intelligent component design.

---
## Wind Resistance Prediction: Efficiency Gains

![width:500px](screenshots/scene_6.jpg)

*   **Traditional vs. AI-Driven Analysis:**
    *   **Traditional:** Labor-intensive, requiring multiple steps and physical prototypes.
    *   **AI-Driven:** Identifies key areas, filters designs, and reduces verification.
*   **AI-Driven Workflow:**
    *   Focuses on optimizing vehicle shape and refining details.
    *   Uses approximately 50 key feature parameters for input.
*   **Benefits:**
    *   Time Efficiency: Reduces verification work.
    *   Early Intervention: Applied during concept design.
    *   Design Optimization: Provides guidelines for features.

---
## Aerodynamic Analysis: User-Friendly Interface

![width:500px](screenshots/scene_7.jpg)

*   **Interface Overview:** A Chinese-language interface for wind resistance (drag coefficient) prediction.
*   **Functionality:**
    *   Allows users to input vehicle parameters.
    *   Provides a one-step analysis using the AI model.
*   **Significance:** Makes complex CFD analysis accessible. Allows rapid testing of aerodynamic properties, and supports rapid design iterations.

---
## AI Project Assistant: Smarter Project Management

![width:500px](screenshots/scene_8.jpg)

*   **AI Project Assistant:** Transforming project management with AI.
    *   Manages multiple projects simultaneously.
    *   Integrates with Work Breakdown Structure (WBS).
    *   Automated reminders and notifications.
    *   Quick filtering of relevant tasks.
*   **Benefits:**
    *   Improved Efficiency.
    *   Systems integration.
    *   Better project monitoring.

---
## Work Item Status: Real-time Insights

![width:500px](screenshots/scene_9.jpg)

*   **Status Tracking & Notification:**
    *   Real-time status tracking with completion percentages.
    *   Progress classification (normal or with issues).
    *   Automated notifications upon work completion.
*   **Proactive Monitoring:**
    *   Initiate progress inquiries anytime.
    *   Consolidated reporting with overall project progress.
*   **Visual Indicators:** Color-coded status indicators and detailed problem descriptions upon hover.

---
## Research Note Assistant: Deeper Understanding

![width:500px](screenshots/scene_10.jpg)

*   **Key Capabilities:**
    *   Converts interviews to text (audio or Word documents).
    *   Organizes content by meaning.
    *   Maps interview segments to questions.
    *   Performs comprehensive content analysis.
*   **Benefits for Researchers:**
    *   Streamlines tasks, freeing researchers for high-level analysis.
    *   Improves work quality and team productivity.
    *   Makes complex information accessible at a glance.

---
## Innovation Through Partnerships

![width:500px](screenshots/scene_11.jpg)

*   **Custom Speech Recognition:**  "Hot words" functionality improves accuracy for technical terms.
*   **Integration with Multiple AI Models:**  Supports various AI models for specialized needs.
*   **Industry Applications:**  Automotive design, manufacturing, and research and development.
*   **NVIDIA Support:** Advanced hardware, access to Omniverse.
*   **Future Direction:** Continued open and collaborative approach.

---
## 3D Visualization in Omniverse: A New Experience

![width:500px](screenshots/scene_12.jpg)

*   **Omniverse for Advanced Visualization:**
    *   Path-tracing real-time rendering for realistic effects.
    *   Dynamic demonstrations for design reviews and marketing.
    *   Hierarchical management and transformation for easy viewpoints.
*   **Interactive Displays:**
    *   Manipulate 3D models from various angles.
    *   Enhances interactivity and intuitiveness of presentations.
    *   Creates a more engaging experience.
*   **Collaboration:**
    *   Multiple team members can edit simultaneously.
    *   Real-time sharing of progress.
    *   Unified environment across design and engineering.

---
## Streamlined Design-to-Visualization

![width:500px](screenshots/scene_13.jpg)

*   **Traditional Bottlenecks Eliminated:** Time-consuming process of exporting to separate rendering software.
*   **Integrated Design Pipeline:**
    *   Real-time changes in Omniverse Composer.
    *   Material selection without application switching.
    *   Performance metrics shown at 75 FPS.
*   **Collaborative Benefits:**
    *   Real-time feedback for all team members.
    *   Connector technology integrates with design tools.

---
## Automotive Styling Accelerated

![width:500px](screenshots/scene_14.jpg)

*   **AI-Powered Transformation:** Simple sketches become finished concept renders.
*   **Previous Design Challenges:** Slow speeds, inefficiencies, distortion problems, and quality issues.
*   **NVIDIA GPU-Accelerated Improvements:** Faster inference, reduced generation time, accurate geometry, and realistic materials.
*   **Streamlined Workflow:**
    *   Rapid generation of designs from sketches.
    *   Intuitive process for visualizing concepts.
    *   Multiple design variations.

---
## AI in Aerodynamics Development: Quantifiable Gains

![width:500px](screenshots/scene_15.jpg)

*   **Efficiency Improvements:** Streamlines aerodynamic development, reducing time and costs.
*   **Quantifiable Gains:** Significant time savings across development stages.
*   **Example: CAS1.0:** Reduced work time by 49%.
*   **Deep Learning Implementation:** AI models developed in collaboration with NVIDIA, achieving prediction accuracy with 3-5% average deviation.

---
## Key Takeaways

*   **NVIDIA Omniverse and AI are transforming automotive design workflows.**
*   **Unified Asset Management (OpenUSD & Nuclear) streamlines data flow.**
*   **RAG technology enables efficient knowledge sharing.**
*   **AI is used in design visualization, CAD, and CFD for efficiency gains.**
*   **Data vectorization converts information into machine-readable formats.**
*   **AI-powered wind resistance prediction significantly reduces development time.**
*   **AI Project Assistant improves project management.**
*   **Research Note Assistant enhances interview analysis.**
*   **Omniverse enhances visualization and collaboration.**
*   **AI accelerates the creative process and improves design quality.**
*   **Significant efficiency gains across development stages are achieved.**