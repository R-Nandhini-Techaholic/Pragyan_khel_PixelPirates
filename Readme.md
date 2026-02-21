# 🏴‍☠️ Pragyan Khel: PixelPirates

An AI-powered **computer vision analytics system** built for the **Pragyan Khel technical challenge**.  
PixelPirates combines **YOLOv8** object detection with a **Streamlit interactive dashboard** to process visual data, track entities, and generate real-time gameplay insights.

# Deployed Link
#https://pixelpiratespragyan.streamlit.app/
---

## 🚀 Key Highlights

- 🎯 **Real-time Object Detection** using YOLOv8 Nano for high FPS inference  
- 📊 **Interactive Streamlit Dashboard** for live configuration and visualization  
- 🧠 **Logic Engine** for geometric reasoning, distance computation, and path estimation  
- 📹 **Live + Uploaded Video Support** for flexible input handling  
- 🖥 **Visual HUD Overlay** with bounding boxes, labels, and tracking paths  
- ⚡ **Lightweight & Modular Design** optimized for fast experimentation  

---

## 🏗 System Architecture

```text
+-------------------------------------------------------+
|                 Streamlit Frontend (app.py)           |
|  [ Upload Video ] [ Live Feed ] [ Confidence Slider ] |
+-------------------------------------------------------+
           |                         |
           v                         v
+-----------------------+     +-------------------------+
|   YOLOv8 Inference    |     |    Logic Engine         |
|   (yolov8n.pt)        |---->|      (/utils/)          |
| Detects: Players,     |     | Calculates: Distances,  |
| Obstacles, Targets    |     | Targets, Optimal Paths  |
+-----------------------+     +-------------------------+
           |                         |
           +------------+------------+
                        |
                        v
+-------------------------------------------------------+
|                Output Visualization                   |
|  (Processed Frames + HUD + Performance Metrics)      |
+-------------------------------------------------------+
```
## Project structure 

```text
Pragyan_khel_PixelPirates/
│
├── app.py                # Streamlit application entry point
├── yolov8n.pt            # YOLOv8 Nano pretrained weights
├── requirements.txt      # Project dependencies
│
├── utils/                # Custom logic modules
│   ├── geometry.py
│   ├── tracking.py
│   └── parser.py
│
└── assets/               # Demo videos, icons, images
```
##  Installation
1️⃣ Clone Repository
git clone https://github.com/R-Nandhini-Techaholic/Pragyan_khel_PixelPirates.git
cd Pragyan_khel_PixelPirates
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run Application
streamlit run app.py
