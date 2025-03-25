---
marp-theme: proposal
title: "Retrieval-Augmented Generation (RAG) for Automotive: Enhancing User Experience and Efficiency"
subtitle: "From Challenges to Solutions: A Deep Dive into Advanced RAG Techniques"
taxonomy: "AI > RAG > Automotive"
---

## Introduction: LLMs in the Business Context
*   Large Language Models (LLMs) offer immense potential.
*   However, LLMs face challenges in specialized business contexts.
*   Introducing Retrieval-Augmented Generation (RAG) to address these limitations.

![width:500px](screenshots/scene_1.jpg)

---

## Challenges of LLMs in Business

*   **Lack of Specialized Knowledge:** LLMs may lack internal company data.
*   **Information Staleness:** LLMs are trained on static data, becoming outdated.
*   **Hallucinations:** LLMs can generate inaccurate or fabricated information.

---

## RAG: The Solution for Automotive Applications
*   RAG connects LLMs to up-to-date, company-specific data.
*   Addresses knowledge gaps, information staleness, and hallucinations.
*   A more practical and cost-effective method for automotive applications.

---

## How RAG Works

*   **Question Input:** User asks a question.
*   **Data Retrieval:** System searches internal data for relevant information.
*   **Information Integration:** Combines question with retrieved data.
*   **Answer Generation:** LLM generates an answer based on the information.

![width:500px](screenshots/scene_2.jpg)

---

## RAG Applications

*   **Internal Employee Systems:** Training, information retrieval.
*   **External Customer Service:** Customer service chatbots.
*   **Developer Tools:** Coding assistants.
*   **Automotive configurators:** Improve accuracy

---

## Challenges of Implementing RAG
*   **Document Parsing:** Handling diverse data formats (Word, PDF, etc.).
*   **Retrieval Stage:** Using vector models trained on general data, and the cost of data creation.
*   **Generation Stage:** Optimizing smaller LLMs for accurate answers.

---

## Boosting RAG Performance: Solutions and Experiments

*   Standard RAG often falls short of enterprise demands.
*   Introducing "DuoMote RAG" (likely a custom implementation).
*   Collaborative development approach with clients.
*   Experimental results on real customer data.

![width:500px](screenshots/scene_3.jpg)

---

## Advanced RAG: Enhancing Document Processing and Retrieval

*   Advanced RAG improves accuracy and handles complex documents.
*   Multimodal PDF Data Extraction Blueprint by NIMO Product team.
*   Object Detection, Table Models, OCR, and a system diagram.

![width:500px](screenshots/scene_4.jpg)

---

## Multimodal PDF Data Extraction Blueprint
*   **Step 1: Element Decomposition:** Breaking down documents.
*   **Step 2: Reading Order Reconstruction:** Arranging elements correctly.
*   **Step 3: Markdown Conversion:** Converting elements to Markdown format.
*   **Step 4: Image and Table Enhancement:** Generating text descriptions.

---

## Enhanced Image-Text Integration
*   Advanced RAG can answer user questions using text and images.
*   Handles both general queries and those related to visual elements.

![width:500px](screenshots/scene_4.jpg)

---

## Enhancing Knowledge Retrieval with Vector Models and Fine-Tuning

*   Vector Model is key to accurate information retrieval.
*   Nemo Retriever Wave Service (NVIDIA) for retrieval.
*   Fine-tuning: Customizing the vector model for specific data.
*   Semi-Automated Data Creation.

![width:500px](screenshots/scene_5.jpg)

---

## Fine-Tuning Data Creation
*   Document Preprocessing, Chunking, and Random Sampling.
*   Prompt Engineering with LLMs.
*   Positive and Negative Trunks (samples).
*   Contrastive Learning Loss.

---

## NVIDIA's Nemo Framework for Fine-Tuning
*   Performance gains after fine-tuning.
*   Significant improvement in "TOP1 hit rate."

![width:500px](screenshots/scene_5.jpg)

---

## Enhancements for Retrieval Performance
*   Contextual Store Retrieval.
*   Semantic Splatter.
*   Query Decomposition.
*   Graph-RAG.

---

## Fine-Tuning and Application of Advanced RAG

*   Dissolution experiments: Measuring optimization impact.
*   Fine-tuning LLMs for specific domains.
*   Two case studies: Customer service chatbots, NVIDIA Triton documentation.

![width:500px](screenshots/scene_6.jpg)

---

## Fine-Tuning Approach
*   **Domain Knowledge Injection.**
*   **Building High-Quality Question-Answer Pairs.**
*   **SFT (Supervised Fine-Tuning)**

---

## Real World Case Studies: Intelligent Customer Service Chatbots

*   Customer Feedback/Question (FBQ) data.
*   De-identified chat logs between the company and its customers.
*   "Continual protruding" and SFT led to improvements in the chatbot's performance.

---

## Real World Case Studies: NVIDIA Triton Technical Documentation Question Answering

*   Developer Documentation.
*   Discussions from the Triton GitHub repository.
*   Questions and answers manually created by team members.
*   Fine-tuning LLMs enhances the performance in answering questions in a highly specific domain.

---

## AI-Powered Automotive Assistance: Enhancing the User Experience
*   Modern cars are complex.
*   Introducing an intelligent "car assistant" or "digital copilot."
*   The car’s system is likely using RAG, as mentioned in the previous frames.

![width:500px](screenshots/scene_7.jpg)

---

## In-Car Assistance
*   **Step 1:** User asks a question.
*   **Step 2:** System understands the question (NLP).
*   **Step 3:** System searches the car's "knowledge base" (RAG).
*   **Step 4:** System provides an answer (voice, on-screen).

---

## Advanced RAG in Automotive: Addressing Needs Across the Lifecycle

*   Automotive industry is complex.
*   RAG addresses needs across R&D, Production/Manufacturing, Marketing, and After-Sales Service.

![width:500px](screenshots/scene_8.jpg)

---

## RAG vs. Traditional Methods
*   Traditional knowledge retrieval (Knowledge Triplet Retrieval) requires manual knowledge extraction.
*   FAQ Retrieval struggles with new questions.

---

## RAG Solution
*   RAG is a smart search engine paired with an AI "answer generator."
*   *How RAG Works*: First, it finds the relevant information from many sources (R**etrieval**). Then, it uses this information to "compose" or **Generate** a concise and accurate answer to the question.
*   Advantages of RAG.
*   Common RAG Frameworks (such as `Lamaindex`)
*   Limitations of simple implementations.

---

## Long Range's Advanced Approach
*   Fine-tuning the Language Model.
*   Using Small Sample Size Techniques.
*   Promt Engineering.
*   Multi-Path Retrieval.
*   Optimized knowledge extraction.
*   Automated extraction and Analysis.

---

## Advanced RAG Optimization: Enhancing Knowledge Retrieval and Generation

*   Multi-Strategy Knowledge Retrieval and Answer Generation.
*   Query Understanding and Contextual Awareness.
*   Optimizing Knowledge Extraction from Standard Pages.
*   Multi-Perspective Question Expansion.
*   Optimizing the Retrieval Module.
*   Large Language Model (LLM) Optimization.

![width:500px](screenshots/scene_9.jpg)

---

## Multi-Strategy Knowledge Retrieval
*   Semantic Search/Vector Search.
*   Concept Words and Title-Based Retrieval.
*   Secondary Recall.

---

## Query Understanding
*   Query Rewriting and Expansion.
*   Adding a Recall Pre-Module.

---

## Optimization of Standard Pages
*   YOLO V8.
*   Extracting information from eleven different kinds of areas to obtain relevant information, and a tree structure to ensure that the relevant information is complete and traceable.

---

## Multi-Perspective Question Expansion
*   The system uses the LLM to simulate various "personas" when rewriting questions, considering perspectives such as car owners, mechanics, and novices.

---

## Retrieval Module Optimization
*   Model Comparison.
*   ACGE Optimization.
*   Data Augmentation.
*   Data Augmentation with Specialized Automotive data.
*   Lexical Expansion.
*   Fault Code Expansion.
*   Contrast Table Loss.
*   Results: Increased recall rate from 70% to 90%.

---

## Large Language Model (LLM) Optimization
*   Leveraging DeepSeek.
*   Continuous Iteration.

---

## Platformization of Advanced Automotive AI Services

*   Standardized platform for AI services.
*   Modular, reusable components.
*   Core services and modules.
*   Modular, incremental development.

![width:500px](screenshots/scene_10.jpg)

---

## Platform Architecture
*   鱼衣服务.
*   知识库服务.
*   生成照会服务.
*   Modular approach to combine in various ways to build different applications.
*   Improve the overall system flexibility and extensibility.

---

## Comparative Performance Summary and Conclusion

*   Comparison of RAG system vs. generic solution.
*   Focus on retrieval accuracy, recall rate, and latency.

![width:500px](screenshots/scene_11.jpg)

---

## Key Performance Metrics
*   **Recall Rate:** 92% overall, 85% for difficult samples.
*   **Latency:** 120 milliseconds or less.
*   **Hallucination Rate ("雾达率"):** Controlled within 5%.

---

## Detail-Oriented Takeaways

*   RAG enhances LLMs by connecting them to up-to-date, specific data.
*   Advanced RAG leverages document processing techniques (like Multimodal PDF Data Extraction Blueprint) to extract information accurately from various sources.
*   NVIDIA's Nemo Retriever Wave service and fine-tuning are used for improved retrieval, and model comparison is essential for choosing a specific model.
*   The presented RAG system outperforms standard solutions in retrieval accuracy, response time, and reducing hallucinations.
*   These advanced RAG technologies drive a better user experience, streamline processes, and improve user satisfaction in the automotive industry, and its platformization promotes collaboration and scalability.
---END MARP DECK---
