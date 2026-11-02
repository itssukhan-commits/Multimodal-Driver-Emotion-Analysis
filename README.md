# Multimodal Transformer-Based Driver Emotion & Behavior Dataset & Replication Suite

This repository provides the complete multimodal dataset and execution pipeline designed to reproduce the findings of the study **"A Multimodal Transformer-Based Framework for Driver Emotion and Behavior Analysis in Complex Traffic Scenarios"** by * Dr. Sajid Ullah Khan and Muhammad Umer Afridi et al.*

---

## 📁 Repository Structure

```
.
├── data/
│   └── driver_multimodal_dataset.csv  # 1,000 synchronized driver-contextual samples
├── src/
│   └── train_and_evaluate.py          # PyTorch Transformer model & 5-fold CV script
├── README.md                          # Study replication guide & documentation
└── requirements.txt                   # Required Python packages
```

---

## 📊 Dataset Schema (`driver_multimodal_dataset.csv`)

The dataset synchronizes three key operational streams across 5 emotion classes (`Calm`, `Stress`, `Anger`, `Fear`, `Fatigue`):

1. **Driver-Centric Features:**
   - `Facial_Stress_Score`: Normalized facial expression stress marker [0, 1]
   - `Gaze_Pitch`, `Gaze_Yaw`: Gaze direction vectors (degrees)
   - `Heart_Rate_BPM`, `HRV_RMSSD_ms`, `EDA_uS`, `EEG_Beta_Power`: Physiological signal indicators

2. **Contextual Traffic Features:**
   - `Scenario`: Driving environment (`Urban`, `Rural`, `Highway`)
   - `Traffic_Event`: Trigger event (`Pedestrian_Crossing`, `Overtaking`, `Accident_Prone`, `Signboard_Interaction`)
   - `Pedestrian_Distance_m`, `Pedestrian_BBox_Area`, `Vehicle_BBox_Area`, `Vehicle_Speed_kmh`

3. **Behavioral Performance Metrics (Table 9 in Study):**
   - `Reaction_Time_ms`: Driver reaction delay in milliseconds
   - `Braking_Accuracy_pct`: Braking precision percentage
   - `Steering_Deviation_deg`: Steering angle drift in degrees
   - `Overtaking_Success_pct`: Successful maneuver execution percentage

---

## 🚀 Execution & Replication Steps

### 1. Requirements Installation
```bash
pip install -r requirements.txt
```

### 2. Run Model Training & Evaluation
```bash
python src/train_and_evaluate.py
```

### 3. Expected Results
The PyTorch transformer implementation evaluates the cross-modal features with 5-fold cross-validation, achieving metrics closely aligned with **Table 8 & Table 10**:
- **Accuracy:** ~91.4%
- **F1-Score:** ~90.4%
