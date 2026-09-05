# 🧪 Universal CSV Analyzer - Liquid Metal Edition

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.1%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9%2B-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Overview

A **production-ready, ultra-fast ML web application** that analyzes any CSV file, auto-detects classification or regression tasks, and trains optimized machine learning models in **under 5 seconds** – all wrapped in a stunning **Liquid Metal glassmorphism UI**.

**Live Demo:** [https://universal-csv-analyzer.onrender.com](https://universal-csv-analyzer.onrender.com)

---

## ✨ Key Features

### 🧠 Ultra-Fast ML Training
- **Auto-drops useless columns** (IDs, names, URLs, near-unique text)
- **Uses OneHotEncoder (sparse) + StandardScaler** for fast preprocessing
- **Automatically selects the best model** based on dataset size:
  - **>50k rows**: SGDClassifier / SGDRegressor (blazing fast)
  - **5k-50k rows**: RandomForest
  - **<5k rows**: SVC / SVR
- **Downsamples large datasets** (>50k rows to 20k) for speed
- **30-second timeout** – no more infinite training
- **Handles any dataset** – classification or regression

### 🎨 Liquid Metal UI
- **Photorealistic metallic sub-cards** (Chrome, Gold, Bronze)
- **Real-time Shader Panel**:
  - Blur, Frostiness, Saturation
  - Tint Alpha + Color
  - Bevel, Specular, 3D Depth
  - Contrast, Brightness
  - **Metal Opacity slider** (transitions between Gel Glass and Metal)
  - Text Color presets + custom picker
- **7 Background images/GIFs** with switcher
- **Fixed unique colors** for section titles
- **Glassmorphism cards** with hover glow + 3D depth
- **Fully responsive** (mobile-friendly)

### 📊 Data Analysis
- Drag & drop CSV upload (up to 64MB)
- Automatic data profiling (dtypes, missing values, unique counts)
- Raw data preview (first 10 rows with alternating colors)
- Auto-detects target column (classification vs regression)
- **Sample datasets** loadable via dropdown
- Metrics: Accuracy, F1, Precision, Recall, R², MSE, MAE
- Charts: Confusion Matrix, ROC Curve, Feature Importance, Actual vs Predicted

---

## 📂 Project Structure
Universal_CSV_Analyzer/
├── app.py # Main Flask application (backend + frontend)
├── Dockerfile # Docker configuration for deployment
├── requirements.txt # Python dependencies
├── sample_data/ # Sample CSV datasets
│ ├── mushrooms.csv
│ └── student_performance_dataset.csv
├── static/
│ ├── images/ # Background images (bg1.jpg - bg7.jpg)
│ └── (HTML, CSS, JS inlined in app.py)
└── README.md # This file


---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Backend language |
| **Flask** | Web framework |
| **scikit-learn** | ML model training |
| **pandas** | Data processing |
| **numpy** | Numerical operations |
| **matplotlib** | Chart generation |
| **OneHotEncoder** | Categorical encoding (fast) |
| **StandardScaler** | Feature scaling |
| **SGDClassifier/Regressor** | Fast training for large datasets |
| **HTML/CSS/JS** | Liquid Metal UI (inline) |
| **Font Awesome 6.4** | Icons |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/olalekanijagbemi-VR/Universal_CSV_Analyzer.git
cd Universal_CSV_Analyzer

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py

# 5. Open in browser
# http://127.0.0.1:5001

☁️ Deployment (Render.com)
This project is configured for free deployment on Render.com.

Push code to GitHub

Go to https://dashboard.render.com

Click "New +" → "Web Service"

Connect your GitHub repository

Select Docker runtime

Choose Free instance type

Click "Deploy Web Service"

📸 Screenshots
(Add screenshots of your app here!)

🎯 Use Cases
Data Scientists: Quickly prototype ML models on any dataset

Students: Learn ML with visual feedback

Business Analysts: Analyze sales data without coding

Educators: Demonstrate classification vs regression

Portfolio Project: Showcase full-stack ML development skills

📄 License
This project is licensed under the MIT License – free for personal and commercial use.

👨‍💻 Author
Olalekan Ijagbemi

🙏 Acknowledgements
Built with ❤️ using Flask, scikit-learn, and custom Liquid Metal CSS

Inspired by Apple's Liquid Glass design language

ML optimization techniques from scikit-learn documentation