# Tool Wear Datasets - Comprehensive Overview

## Dataset Comparison Table with Vibration & Acoustic Emission Details

| ID | Dataset Name | Process Type | Link(s) | **Vibration Data Available** | **Acoustic Emission Available** | Tool Wear Suitability | Detailed Sensor Specifications | Key Features |
|---|---|---|---|---|---|---|---|---|
| **00** | **NASA Milling BaseData** | Milling | https://data.nasa.gov/dataset/milling-wear | ✅ **YES**<br>- Piezoelectric accelerometer<br>- Sampling details not specified | ✅ **YES**<br>- AE sensor included<br>- Frequency range not specified | Good to go for tool wear | **Vibration:** Piezoelectric accelerometer<br>**AE:** Standard AE sensor<br>**Force:** Kistler 3-axis dynamometer<br>**Wear:** Optical microscope measurement (VB in 10^-3 mm) | - Stainless steel (HRC52) workpieces<br>- 10,400 rpm spindle speed<br>- Three-flute ball nose tungsten carbide cutter |
| **01** | **Multi-Sensor CNC Tool Wear** | CNC Milling | https://www.kaggle.com/datasets/ziya07/multi-sensor-cnc-tool-wear-dataset<br>http://www.phmsociety.org/forum/583 | ✅ **YES**<br>- Multi-axis vibration sensors<br>- Details not fully specified | ⚠️ **UNCLEAR**<br>- May include AE<br>- Not explicitly confirmed | Good to go for tool wear | **Vibration:** Multi-axis accelerometers<br>**Force:** Cutting force sensors<br>**Other:** Current sensors, various CNC monitoring sensors<br>**Wear:** Tool wear progression tracking | - Variational CNC machining data<br>- Time-series sensor data format<br>- Industrial CNC machine setup |
| **02** | **PHM Data Challenge 2010** | Milling | https://www.kaggle.com/datasets/rabahba/phm-data-challenge-2010 | ✅ **YES - EXCELLENT**<br>- **3-axis vibration (X,Y,Z)**<br>- **Units: g (acceleration)**<br>- **50 kHz sampling rate**<br>- Piezoelectric accelerometers | ✅ **YES - EXCELLENT**<br>- **AE-RMS in Volts**<br>- **50 kHz sampling rate**<br>- **Column 7 in 7-channel data**<br>- Standard AE sensor setup | **Gold Standard** - Large Dataset, Base data | **Vibration:** 3x piezoelectric accelerometers (X,Y,Z axes, g units)<br>**AE:** AE-RMS sensor (Voltage output)<br>**Force:** Kistler 3-component dynamometer (X,Y,Z, Newtons)<br>**Sampling:** 50 kHz/channel all sensors<br>**Wear:** Optical microscope (VB in 10^-3 mm) | - **7-channel CSV format**<br>- **315 cuts per cutter**<br>- **6 cutters (C1-C6), 3 labeled**<br>- High-speed CNC (Röders Tech RFM760)<br>- **RUL estimation focus** |
| **03** | **Multi-Sensor Tool Wear Monitoring** | Milling | https://www.kaggle.com/datasets/programmer3/multi-sensor-tool-wear-monitoring-dataset | ✅ **YES - EXCELLENT**<br>- **3-axis vibration sensors**<br>- Accelerometers for real-time monitoring<br>- Time-series vibration data | ✅ **YES - EXCELLENT**<br>- **Acoustic emission sensors**<br>- **Vibration and AE combined focus**<br>- Multi-sensor fusion approach | **Excellent** for Vibration & AE studies | **Vibration:** 3-axis accelerometers for multi-sensor monitoring<br>**AE:** Dedicated acoustic emission sensors<br>**Force:** Force measurement systems<br>**Current:** Machine current sensors<br>**Integration:** Multi-sensor data fusion architecture | - **Specifically designed for vibration/AE analysis**<br>- Real-time monitoring capability<br>- Multi-sensor data fusion approach<br>- Predictive maintenance focus |
| **04** | **IoT-Integrated PM** | Simulated | https://www.kaggle.com/datasets/ziya07/iot-integrated-predictive-maintenance-dataset | ❌ **NO/UNCLEAR**<br>- Generic IoT sensors<br>- No specific vibration focus<br>- Simulated data | ❌ **NO**<br>- No acoustic emission data<br>- General IoT monitoring only | **⚠️ CRITICAL USAGE** - simulated data, no tool definition | **Simulated Sensors:** Generic IoT equipment monitoring<br>**No Specific:** Vibration or AE sensors<br>**Focus:** General machine failure prediction<br>**⚠️ Warning:** No real machining data | - **⚠️ Simulated data only**<br>- General machine failure prediction<br>- **No specific tool wear definition**<br>- **No vibration/AE data** |
| **05** | **Intelligent Manufacturing Dataset** | Simulated | https://www.kaggle.com/datasets/ziya07/intelligent-manufacturing-dataset | ❌ **NO/UNCLEAR**<br>- General production sensors<br>- No machining-specific vibration<br>- Simulated environment | ❌ **NO**<br>- No acoustic emission monitoring<br>- Manufacturing efficiency focus | **⚠️ CRITICAL USAGE** - simulated data, no tool definition | **Simulated Sensors:** Network and production monitoring<br>**No Machining:** No vibration or AE sensors<br>**Focus:** Manufacturing efficiency analysis<br>**⚠️ Warning:** No real tool wear data | - **⚠️ Simulated data only**<br>- **No specific tool wear definition**<br>- Focus on manufacturing efficiency<br>- **No vibration/AE capabilities** |
| **06** | **IoT-based Equipment Fault Prediction** | Transfer | https://www.kaggle.com/datasets/programmer3/iot-based-equipment-fault-prediction-dataset | ❌ **UNCLEAR**<br>- General equipment monitoring<br>- Unclear if vibration included<br>- No machining focus | ❌ **NO/UNCLEAR**<br>- No specific AE mention<br>- General fault prediction only | **⚠️ NOT RECOMMENDED** - unclear tool wear definition | **General:** IoT-based equipment monitoring<br>**Unclear:** Specific sensor types not defined<br>**Problem:** No clear tool wear methodology<br>**⚠️ Warning:** Unclear machining context | - **⚠️ Unclear tool wear definition**<br>- **⚠️ Unclear which tool is monitored**<br>- General equipment fault focus<br>- **Not suitable for vibration/AE research** |
| **07** | **Multivariate Time Series Milling** | Milling | https://data.mendeley.com/datasets/zpxs87bjt8/3<br>https://www.sciencedirect.com/science/article/pii/S2352340923006741 | ⚠️ **UNCLEAR**<br>- Process forces available (25 kHz)<br>- **No explicit vibration sensors mentioned**<br>- Machine control data (500 Hz) | ❌ **NO**<br>- No acoustic emission sensors<br>- Focus on force and machine data<br>- Missing AE component | Good for force analysis, **Limited for vibration/AE** | **Forces:** Dynamometer (25 kHz sampling)<br>**Machine Data:** Spindle/feed drive forces (500 Hz)<br>**Control:** Position deviation monitoring<br>**Missing:** Dedicated vibration and AE sensors<br>**Wear:** Digital microscope (VB up to 150 μm) | - **9 end milling cutters** lifecycle<br>- **3 different 5-axis machines**<br>- **6,418 labeled files**<br>- **⚠️ Limited vibration/AE data** |

## Enhanced Analysis for Vibration & Acoustic Emission Research

### **🎯 TOP RECOMMENDATIONS FOR VIBRATION & ACOUSTIC EMISSION RESEARCH:**

#### **Tier 1 - Excellent for Vibration/AE Studies:**
1. **PHM Data Challenge 2010** 🥇
   - **✅ Complete 7-channel data** (3 force + 3 vibration + 1 AE)
   - **✅ 50 kHz sampling rate** for all sensors
   - **✅ Industry standard dataset** with extensive research validation
   - **✅ Clear data format:** CSV with defined units (g for vibration, V for AE-RMS)

2. **Multi-Sensor Tool Wear Monitoring** 🥈  
   - **✅ Specifically designed** for vibration and acoustic emission analysis
   - **✅ 3-axis vibration + dedicated AE sensors**
   - **✅ Multi-sensor fusion focus**
   - **✅ Real-time monitoring approach**

#### **Tier 2 - Good with Limitations:**
3. **NASA Milling BaseData** 🥉
   - **✅ Both vibration and AE sensors** included
   - **⚠️ Limited technical specifications** available
   - **✅ Well-documented** experimental setup
   - **✅ Good baseline** for comparison studies

#### **Tier 3 - Limited Vibration/AE Capabilities:**
4. **Multi-Sensor CNC Tool Wear** 
   - **✅ Vibration data** confirmed
   - **⚠️ AE availability unclear**
   - **⚠️ Limited sensor specifications**

5. **Multivariate Time Series Milling**
   - **⚠️ No dedicated vibration sensors** mentioned
   - **❌ No acoustic emission data**
   - **✅ High-quality force data** (25 kHz)
   - **Best for:** Force-based analysis, not vibration/AE

#### **❌ NOT RECOMMENDED for Vibration/AE Research:**
- **Datasets 04, 05, 06**: No real machining vibration/AE data, simulated/unclear sources

---

### **🔊 ACOUSTIC EMISSION TECHNICAL SPECIFICATIONS:**

#### **Frequency Ranges in Machining:**
- **General AE Range**: 100 kHz - 1 MHz (typical sensor bandwidth: 100-900 kHz)
- **Tool Wear Indicators**: Up to 350 kHz (increases with wear, then saturates)
- **Plastic Deformation**: 50-100 kHz, 51-471 kHz range
- **Crack Propagation/Coating Damage**: Higher frequency range (>200 kHz)
- **Background Noise Avoidance**: >100 kHz (machine noise typically <100 kHz)

#### **AE Sensor Mounting Locations:**
- **Spindle Housing**: Most common for milling applications
- **Tool Holder**: Close to cutting zone for maximum sensitivity
- **Workpiece**: Direct contact for turning applications
- **Wireless Transmission**: For rotating tool applications

#### **Signal Processing:**
- **AE-RMS (Root Mean Square)**: Most common measurement (voltage output)
- **Frequency Analysis**: FFT, STFT, Wavelet transforms
- **Filtering**: High-pass filters (>100 kHz) to eliminate machine noise
- **Correlation**: Tool wear increases AE energy until saturation point

---

### **📳 VIBRATION MEASUREMENT SPECIFICATIONS:**

#### **Sensor Types & Mounting:**
- **Accelerometers**: Piezoelectric, 3-axis measurement (X,Y,Z)
- **Units**: Acceleration (g), velocity (mm/s), displacement (μm)
- **Mounting Locations**: Machine tool structure, spindle, tool holder, workpiece
- **Coupling**: Magnetic, adhesive, stud-mounted for permanent installation

#### **Sampling Rates & Frequency Response:**
- **High-Speed Applications**: 10-50 kHz (PHM 2010: 50 kHz)
- **Standard Monitoring**: 25 kHz (common for vibration analysis)
- **Machine Control**: 0.5-2 kHz (sufficient for low-frequency monitoring)
- **Frequency Range**: DC to 10-20 kHz for most machining applications

#### **Vibration Analysis Features:**
- **Time Domain**: RMS, peak values, crest factor, kurtosis
- **Frequency Domain**: FFT, power spectral density, frequency peaks
- **Time-Frequency**: STFT, wavelet analysis, envelope analysis
- **Machine Learning**: Direct signal input to CNN/LSTM models

---

### **⚠️ CRITICAL USAGE WARNINGS:**

#### **Datasets to AVOID for Vibration/AE Research:**
- **Datasets 04, 05, 06**: 
  - ❌ **No real machining data** (simulated/unclear)
  - ❌ **No vibration/AE sensors**
  - ❌ **Undefined tool wear methodology**
  - ❌ **Not suitable** for sensor-based research

#### **Data Quality Considerations:**
- **Sampling Rate**: Ensure >20 kHz for meaningful AE analysis
- **Sensor Calibration**: Check for calibration data and sensor specifications
- **Background Noise**: Verify noise levels and filtering methodology
- **Synchronization**: Ensure vibration and AE data are time-synchronized

---

### **🔧 SENSOR TECHNOLOGY SUMMARY:**
- **Force/Cutting Forces**: Dynamometers with 3-component measurement (X,Y,Z directions)
- **Vibration**: 3-axis accelerometers, piezoelectric sensors, 10-50 kHz sampling
- **Acoustic Emission**: Wideband sensors (100-900 kHz), AE-RMS measurement
- **Tool Wear Measurement**: Optical microscopes, digital microscopes (offline measurement)

### **📊 COMMON TOOL WEAR METRICS:**
- **VB (Flank Wear)**: Most common metric, measured in micrometers (μm) or 10^-3 mm
- **Tool Life Criterion**: Typically VB ≈ 150 μm based on ISO 3685:1993 standard
- **Measurement Methods**: Offline optical microscopy (most accurate) vs online indirect estimation

### **🏭 MATERIALS & MACHINING CONDITIONS:**
- **Materials**: Stainless steel, cast iron, Ti6Al4V (aerospace applications)
- **Tools**: Tungsten carbide cutters, TiN-TiAlN coated tools, 3-4 flute end mills
- **Processes**: High-speed milling, dry/wet cutting conditions, various cutting parameters