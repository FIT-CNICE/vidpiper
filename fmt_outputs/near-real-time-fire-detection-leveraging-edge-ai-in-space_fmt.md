---
marp-theme: proposal
title: Near Real-Time Fire Detection Leveraging Edge AI in Space
subtitle: Technical Summary for General Audiences
taxonomy: "Wildfire > Detection > Satellite"
---
## OroraTech: Satellite-Based Monitoring

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_1.jpg)

*   **OroraTech Overview:** A space technology company founded in 2018, focused on satellite-based monitoring.
*   **Key Capabilities:** Patented thermal cameras, satellite network, AI-driven analytics, and data delivery systems.
*   **Assets:** 27 satellite data sources, 2 operational satellites.
*   **Recent Developments:** Significant contracts and funding secured.
*   **Focus:** Using NVIDIA hardware for wildfire monitoring.

---
## Wildfire Monitoring: Situational Awareness

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_2.jpg)

*   **Greek Wildfire Case Study:** Illustrates the use of satellite technology in firefighting.
*   **Early Detection:** Satellite detected the fire when it was small.
*   **Real-Time Intelligence:** Simulation capabilities to forecast fire progression.
*   **Resource Allocation:** Satellite intelligence helps firefighting agencies allocate resources.
*   **Critical Value:** Provides data when traditional methods are unavailable (e.g., LA wildfire with strong winds).

---
## FOREST-2: Advanced Satellite Technology

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_3.jpg)

*   **Afternoon Fire Detection Gap:** Many fires ignite in the afternoon, when current satellite coverage is limited.
*   **FOREST-2 Specifications:**
    *   Launched June 2023.
    *   410 km coverage with 200m resolution.
    *   Detects fires as small as 4x4 meters.
    *   Three thermal channels.
    *   On-board GPU for processing.
    *   500 km Sun-Synchronous Orbit.
    *   1.9 day revisit time.
*   **Expanding Constellation:** Plans to deploy more satellites for higher revisit rates.

---
## Temporal Coverage and Latency Reduction

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_4.jpg)

*   **Key Detection Factors:** Temporal coverage (revisit frequency) and data latency (delay).
*   **Strategic Approach:** Satellites launched in formations to monitor every location twice a day.
*   **Thermal Imaging:** Detects heat, unlike human vision (optical range).
*   **Temperature Visualization:** Example from Ghana, showing rivers as darker and land as brighter areas.
*   **Advantages Over Optical:** Can see through smoke to detect active fires.

---
## Optimizing Wildfire Detection: Satellite Bands

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_5.jpg)

*   **Electromagnetic Spectrum:** Focus on specific wavelength bands for optimal wildfire detection.
*   **MWIR (Medium Wave Infrared):** Around 3.8 μm, for active fire detection and Fire Radiative Power measurement.
*   **LWIR (Long Wave Infrared):** In the 10.0-11.0 μm range, for Land Surface Temperature (LST) and fire risk assessment.
*   **Band Alignment:** Aligned with public satellite missions for improved data quality.
*   **Data Processing Hierarchy:**
    *   Level 1: Raw data
    *   Level 2: Active Fire product and Land Surface Temperature (LST)

---
## On-board Data Processing

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_6.jpg)

*   **Technical Specs:** 200-meter GSD, 4x4-meter sensitivity, 8 Kelvin accuracy in MWIR, 3 Kelvin in LWIR, 400 x 450 km image coverage, and twice-daily revisit.
*   **Data Collection:** Three spectral bands record data in three strips per image.
*   **Scanning Process:** Progressive scanning ensures all points on Earth are observed in all bands.
*   **L1 Pipeline:** Transforms raw data into usable information with various corrections.

---
## L1 Pipeline and CUDA

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_7.jpg)

*   **Data Processing Workflow:** Raw sensor data to application-ready data through L0 to L2 stages.
*   **Frame-Based vs. Integrated Processing:** Operations per frame, then integration in L1C stage.
*   **Infrared Band Integration Challenge:** Combining data from different bands across different frames.
*   **Resampling Process:** Transforms frame data into a unified grid to solve the integration challenge.
*   **CUDA Optimization:** Develops a specialized processing framework and CUDA implementation for GPU execution.

---
## CUDA Acceleration

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_8.jpg)

*   **Hybrid Architecture:** CPU-GPU pipeline for efficiency.
*   **Strategic GPU Optimization:** CUDA used for L1C and L2 processing.
*   **Parallel Processing:** Simultaneously processes different frames on CPU and GPU to maximize throughput.
*   **Performance and Detection:** Faster processing leads to more frequent observations and earlier detection.
*   **NVIDIA Jetson:** Ideal balance of performance and power consumption for satellite use.

---
## Near Real-Time Wildfire Detection

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_9.jpg)

*   **Complete Fire Detection Pipeline:** From image capture to user notification.
*   **End-to-End Process:**
    1.  Image Acquisition
    2.  Pre-processing
    3.  Fire Detection using AI
    4.  Downlink
    5.  User Notification (under 10 minutes)
*   **Key Innovations:** On-board processing using NVIDIA Jetson, data minimization, and dedicated infrastructure.
*   **Practical Impact:** Reduces detection time from hours to minutes, enabling quicker response.

---
## ML-based Active Fire Detection

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_10.jpg)

*   **Transformer vs. CNN:** Comparing neural network architectures.
*   **Segformer Architecture:** Transformer-based, from NVIDIA, lightweight enough for on-orbit inference, and 3.7 million parameters.
*   **Comparison with CNN:** Compares with ResNet50, MobileNet, and EfficientNet.
*   **Dataset Characteristics:** Custom dataset, globally distributed, with class imbalance.
*   **Technical Significance:** Smaller parameter size is critical for satellite deployment.

---
## Performance Results

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_11.jpg)

*   **Model Performance:** Transformer outperforms CNNs; reached an 88% F1 score.
*   **Data Augmentation:** Techniques used to improve training.
*   **Cloud Detection Importance:** Cloud cover is high, and cloud detection helps overcome downlink limitations and prevent false positives.
*   **Technical Application:** Landsat 7 example showing cloud detection challenges.
*   **Dataset Development:** Cloud detection also relies on a small global dataset.

---
## Cloud Detection: Model Comparison

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_12.jpg)

*   **Qualitative Results:** Comparing different cloud detection techniques.
*   **Detection Model Comparison:** Thresholding, XGBoost, MobileNet, and MobileNet-Joint.
*   **Key Performance Insights:** Deep learning models outperform thresholding in challenging scenarios.
*   **Performance Metrics:** ML models often achieving >90% accuracy.
*   **Key Takeaway:** Advanced ML provides significant value for cloud detection.

---
## Cloud Detection: On-board Implementation

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_13.jpg)

*   **On-board Implementation:** Cloud detection directly on the satellite.
*   **Performance and Improvements:** 87% accuracy, 81% macro F1 score.
*   **Technical Optimization:** Runs in less than 5 seconds, GPU acceleration using TensorRT, ONNX format.
*   **Comparative Analysis:** Outperforms thresholding.
*   **Real-world Application:** Monitors events such as the Palisade fire.

---
## Active Fire Validation

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_14.jpg)

*   **Multi-Layered Validation System:** Validates fire detections.
*   **Key Validation Metrics:**
    1.  Max FWI
    2.  Max FRP
    3.  Max nFRP
    4.  Landcover Analysis
    5.  GEO Persistence
    6.  Number of Independent Satellites
    7.  Sensor and Algorithm Combinations
*   **Integrated Validation:** Combines multiple data sources.
*   **Benefits:** Reduce false positives and improve reliability.

---
## Short-Term Fire Risk Prediction

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_15.jpg)

*   **Model Comparison:** Comparing fire risk assessment approaches.
*   **Fire Weather Index (FWI):** Industry standard model with low performance metrics.
*   **OroraTech's AI Model:** Significantly improved performance with up to 44x more accurate fire hazard prediction.
*   **Technical Approach:** Accurate fuel modeling, enhanced wind forecasts, and historical data archive.
*   **Multi-Temporal Prediction:** Hourly, daily, and seasonal fire prediction capabilities are being developed.

---
## Thermal Data for Urban Heat

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_16.jpg)

*   **Land Surface Temperature Monitoring:** Focus on land surface temperature.
*   **Technical Specifications:** 200-meter resolution, up to 3 Kelvin accuracy, 3-hour data delivery, and twice-daily coverage.
*   **Urban Heat Island Detection:** Thermal imagery examples from Los Angeles and Linyares, Brazil.
*   **Practical Applications:** Inform urban planning, land use decisions, and public health.

---
## Thermal Data Validation & Vision

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_17.jpg)

*   **Data Validation:** Validating LST data with ground-based measurements.
*   **Future Vision:** Create a "thermal digital twin" with 30-minute global temperature updates.
*   **Enhancements Through AI:** Improved spatial resolution through AI.
*   **Significance for Climate Science:** Provides critical data for climate change understanding.

---
## Thermal Data Applications

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_18.jpg)

*   **Additional Applications:** Sea Surface Temperature monitoring.
*   **30-Minute Refresh Rate:** Potential of a half-hourly global thermal data update.
*   **Data Infrastructure:** Ingesting over 5 terabytes of data daily.
*   **Customer Access Options:** API access and customizable visualization platform.
*   **Value:** Provides comprehensive thermal Earth observation data.

---
## NVIDIA Developer Program

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_19.jpg)

*   **NVIDIA Developer Program:** Resources for developers.
*   **Key Resources:** Free tools, expert support, and application focus.
*   **Application Examples:** Applications for data centers, healthcare, city monitoring, and more.
*   **Access:** developer.nvidia.com/join

---
## NVIDIA Connect Program

![width:500px](home/fit-sizhe/dwhelper/gtc/sum_outputs/near-real-time-fire-detection-leveraging-edge-ai-in-space_output/screenshots/scene_20.jpg)

*   **NVIDIA Connect Program:** For Independent Software Vendors (ISVs).
*   **Key Benefits:** Developer resources, technical training, and preferred pricing.
*   **Visual Representation:** Cubes showing AI diagrams, SDKs and hardware.
*   **Access:** nvidia.com/connect-program

---
## Detail-Oriented Takeaways

*   OroraTech leverages a multi-layered approach, from specialized satellites to on-board processing and advanced AI models, to detect and validate wildfires in near real-time.
*   The system uses NVIDIA Jetson Xavier GPUs to optimize the data processing pipeline, achieving low latency and high accuracy, crucial for timely fire response.
*   Innovative machine learning techniques, including Segformer for active fire detection and advanced cloud detection models, demonstrate superior performance over conventional methods.
*   Their system incorporates a multi-source validation system that uses several factors for determining fire confidence level, including Max FWI, FRP, and others.
*   The company’s AI-driven fire risk prediction model offers significant improvements over industry-standard tools, providing more accurate short-term hazard forecasts.
*   Beyond wildfire detection, OroraTech’s technology monitors land and sea surface temperatures. The vision of providing global temperature updates every 30 minutes could revolutionize urban planning, climate science, and environmental monitoring.
*   The company provides data access via APIs and custom visualization platforms, and is looking to partner with developers.