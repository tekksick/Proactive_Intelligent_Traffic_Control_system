# 🚦 Proactive Intelligent Traffic Signal Optimization  
### **Deep Learning + LSTM Forecasting + Model Predictive Control (MPC)**  
**Authors:** Suhana Shaik, Kambam Sai Ruchitha, Tejaswini Raj Koneti, Anamika Mangore, Ayan Mondal (Senior Member, IEEE)

---

## Overview

Urban traffic congestion is a growing challenge that leads to significant delays, fuel consumption, and environmental pollution. Traditional fixed-time or actuated controllers are **reactive** — they respond only after congestion forms.

This project proposes a **Proactive Intelligent Traffic Control System (PITCS)** combining:

- **YOLOv12** for real-time vehicle detection  
- **Bidirectional LSTM** for short-term traffic forecasting  
- **Model Predictive Control (MPC)** for optimal signal timing  

This system **predicts congestion** *before* it forms and optimizes cycle timings in advance, reducing queue length, delay, and unnecessary signal switching.

---

## Problem Statement Diagram

> <img width="345" height="652" alt="image" src="https://github.com/user-attachments/assets/4d959b83-4cd5-4d29-8423-2057a655afdd" />
 
Place your image file inside the repository and reference it like this:

```markdown
![Problem Statement](assets/problem_statement.png)

System Architecture
          ┌──────────────────┐
          │  YOLOv12 Detector │
          └─────────▲────────┘
                    │ Real-time Vehicle Counts
                    ▼
          ┌──────────────────┐
          │ BiLSTM Forecast  │───► Future Traffic Prediction
          └─────────▲────────┘
                    │
                    ▼
          ┌──────────────────┐
          │      MPC         │───► Optimal Signal Timings
          └──────────────────┘

📂 Repository Structure
📁 Proactive_Intelligent_Traffic_Control_System
│
├── 📁 YOLO_Vehicle_Detection
│   ├── training_scripts/
│   ├── weights/
│   └── inference/
│
├── 📁 Traffic_Forecasting_LSTM
│   ├── dataset/
│   ├── preprocessing/
│   ├── model_training/
│   └── multi_lane_forecasting/
│
├── 📁 MPC_Controller
│   ├── optimization/
│   ├── queue_models/
│   ├── constraints/
│   └── simulations/
│
├── 📁 Results
│   ├── YOLO_evaluation/
│   ├── Forecasting_graphs/
│   └── MPC_vs_FixedTime/
│
└── README.md

Module Breakdown
1️⃣ YOLOv12 Vehicle Detection

Trained on top-view traffic dataset

YOLOv12 chosen for highest accuracy

Key improvements:

Area Attention (A²)

R-ELAN modules

Better spatial-context modeling

Performance Comparison

Model	Precision	Recall	mAP50
YOLOv8	0.929	0.918	0.975
YOLOv11	0.917	0.939	0.974
YOLOv12	0.932	0.915	0.977
2️⃣ Bidirectional LSTM Traffic Forecasting

Dataset: I-94 Traffic Volume
Prediction horizon: 2 hours (5-min resolution)

Model	MAE	RMSE
MODEL-1 (Stacked LSTM)	276.75	449.40
MODEL-2 (Bi-LSTM)	236.02	431.39
MODEL-3 (Deep LSTM)	241.64	437.88

✔ Bi-LSTM demonstrated the best forecasting performance.

3️⃣ Model Predictive Control (MPC)

MPC uses:

Store-and-Forward (SF) queue model

LSTM-predicted inflows

Safety/time constraints

Switching penalties to reduce cycle disturbance

MPC vs. Fixed-Time Results

Metric	MPC	Fixed-Time
Avg Queue Length	63.60	88.20
Total Delay (veh-sec)	167,493	225,799
Phase Switches	3.5	13.5

✔ MPC significantly reduces congestion and unnecessary switching.

⚙ Installation & Setup
1. Clone the Repository
git clone https://github.com/tekksick/Proactive_Intelligent_Traffic_Control_system.git
cd Proactive_Intelligent_Traffic_Control_system

2. Install Dependencies
pip install -r requirements.txt

▶️ Running the Modules
Run YOLOv12 Inference
python YOLO_Vehicle_Detection/inference/detect.py --weights yolov12.pt --source sample_video.mp4

Run LSTM Training
python Traffic_Forecasting_LSTM/model_training/train_lstm.py

Run MPC Controller
python MPC_Controller/run_mpc.py

📈 Results Summary

YOLOv12 achieves state-of-the-art detection accuracy

Bi-LSTM provides superior multi-step forecasting

MPC reduces queue length, waiting time, and switching frequency

End-to-end system outperforms traditional controllers clearly

🧭 Future Enhancements

DeepSORT + YOLO for multi-object tracking

Max-Pressure control integration with MPC

Deployment on Jetson Nano or edge hardware

Multi-intersection city-scale simulations

V2I and connected-vehicle integration

📚 Citation

If you use this work, cite the authors:

<Will be updated after publication>

⭐ Acknowledgments

This research is guided by Dr. Ayan Mondal (Senior Member, IEEE)
and contributes to ongoing Intelligent Transportation Systems (ITS) advancements.
