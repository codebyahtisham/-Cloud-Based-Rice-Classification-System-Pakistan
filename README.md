# 🌾 Cloud-Based Rice Classification Using Machine Learning & Image Processing

> An automated, cloud-deployed rice classification system that identifies **7 Pakistani rice varieties** from scanner and mobile images using **49 handcrafted features** and **XGBoost** — built in collaboration with **Alkaram Rice Engineering (PVT) Ltd.** and the **Centre for AI & Big Data, Namal University**.

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Domain](https://img.shields.io/badge/Domain-AI%20%7C%20AgriTech-blue)
![Type](https://img.shields.io/badge/Type-Final%20Year%20Project-orange)
![Model](https://img.shields.io/badge/Model-XGBoost-purple)
![Accuracy](https://img.shields.io/badge/Scanner%20Accuracy-88.86%25-success)
![Accuracy](https://img.shields.io/badge/Mobile%20Accuracy-80.54%25-yellow)
![Dataset](https://img.shields.io/badge/Dataset-7000%2B%20Images-informational)
![Cloud](https://img.shields.io/badge/Deployed-Namal%20HPC%20Cloud-lightgrey)

---

## 📌 Overview

Rice is Pakistan's second-largest export commodity, yet variety identification still relies on **slow, error-prone manual inspection**. This project delivers a **production-ready, cloud-based web application** that automates rice variety classification using computer vision and machine learning.

We built the **entire pipeline from scratch** — from physically collecting and digitizing 7,000+ rice grain images at an industry research lab, to extracting 49 visual features per grain, training an XGBoost classifier, and deploying a Streamlit web app on Namal's HPC Cloud infrastructure. The system supports **two input modes**: a flatbed scanner for industrial-grade lab use and a mobile camera with a custom calibration grid for on-field deployment by farmers.

This is a **real-world industrial project** developed over one year in partnership with a leading rice engineering company, addressing actual quality control challenges in Pakistan's rice export industry.

---

## 🎯 Problem Statement

Pakistan's rice industry faces critical challenges in variety identification:

- **Manual inspection** is slow, subjective, and costly
- Human judgment varies person-to-person, leading to **quality inconsistencies**
- Misclassification causes **financial losses** and reduced trust in global markets
- No affordable, accessible automated solution existed for Pakistani rice varieties

**Our solution:** An AI-powered system that classifies rice varieties from a single image — accessible from anywhere via the cloud.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT MODES                              │
│                                                                 │
│   ┌──────────────┐              ┌──────────────────────┐        │
│   │ 📷 Scanner   │              │ 📱 Mobile + Grid     │        │
│   │ Canon LiDE   │              │ Red 6.35mm grid      │        │
│   │ 600 DPI      │              │ for calibration      │        │
│   └──────┬───────┘              └──────────┬───────────┘        │
│          │                                 │                    │
│          ▼                                 ▼                    │
│   ┌──────────────┐              ┌──────────────────────┐        │
│   │ Otsu Thresh  │              │ HSV Grid Detection   │        │
│   │ + Watershed  │              │ + Inpainting Removal │        │
│   └──────┬───────┘              └──────────┬───────────┘        │
│          │                                 │                    │
│          └──────────────┬──────────────────┘                    │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  FEATURE EXTRACTION │                            │
│              │  49 Features/Grain  │                            │
│              │                     │                            │
│              │ 13 Morphological    │                            │
│              │  9 Color (RGB)      │                            │
│              │ 26 Texture (GLCM+LBP)│                          │
│              │  1 Edge             │                            │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   XGBoost Model     │                            │
│              │  300 trees, depth 7 │                            │
│              │  lr=0.1, L1/L2 reg  │                            │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  CLASSIFICATION     │                            │
│              │  + CSV Report       │                            │
│              │  + Labeled Image    │                            │
│              └─────────────────────┘                            │
│                                                                 │
│   ☁️ Deployed on Namal HPC Cloud via Streamlit                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌾 Supported Rice Varieties

Seven pure Pakistani rice varieties sourced from **Haji Muhammad Rice Mill** research lab:

| # | Variety | Length (mm) | Width (mm) | Color | Size |
|---|---------|:-----------:|:----------:|-------|------|
| 1 | **Super Brown** | 7.49 | 3.18 | Brown | Long |
| 2 | **C9 Sella** | 6.85 | 1.86 | Golden Yellow | Long |
| 3 | **PK 386** | 5.93 | 2.79 | White | Long |
| 4 | **Super 109** | 6.98 | 2.49 | White | Long |
| 5 | **Super Silky** | 8.00 | 1.98 | White | Extra Long |
| 6 | **Super White** | 7.28 | 3.51 | White | Extra Long |
| 7 | **Supri Sella** | 6.68 | 2.92 | Golden Yellow | Long |

---

## 🔬 Methodology

### 1 — Dataset Creation (Novel Contribution)

One of the most significant contributions of this project was the **digitization of Pakistani rice varieties** — no such labeled dataset previously existed.

- **Scanner Dataset:** 5,000 images captured at 600 DPI using Canon LiDE 300 with black background under controlled lighting
- **Mobile Dataset:** 1,000+ images captured on Android smartphones using a custom **red grid paper (0.25″ / 6.35mm squares)** for spatial calibration
- **Total:** 7,000+ labeled grain images across 7 varieties

### 2 — Feature Engineering (49 Features per Grain)

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  MORPHOLOGICAL   │     │      COLOR       │     │     TEXTURE      │
│  (13 features)   │     │   (9 features)   │     │  (26 features)   │
├─────────────────┤     ├──────────────────┤     ├──────────────────┤
│ Length, Width    │     │ Mean R/G/B       │     │ GLCM Contrast    │
│ Area, Perimeter  │     │ Std Dev R/G/B    │     │ GLCM Energy      │
│ Aspect Ratio     │     │ Skewness R/G/B   │     │ GLCM Entropy     │
│ Circularity      │     │                  │     │ LBP Histogram    │
│ Compactness      │     │                  │     │ (16 bins)        │
│ Eccentricity     │     │                  │     │                  │
│ Extent           │     │                  │     │                  │
│ Major/Minor Axis │     │                  │     │                  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                              + 1 Edge Feature
```

### 3 — Model Selection: Why XGBoost over Deep Learning?

| Criteria | XGBoost (Chosen) | CNN / DNN |
|----------|:----------------:|:---------:|
| Dataset size needed | Small ✅ | Large ❌ |
| Training time | Minutes ✅ | Hours ❌ |
| Interpretability | Full (SHAP) ✅ | Black-box ❌ |
| GPU required | No ✅ | Yes ❌ |
| Cloud deployment | Lightweight ✅ | Heavy ❌ |
| Industry trust | Explainable ✅ | Opaque ❌ |

**Model Hyperparameters:** 300 boosting rounds · max depth 7 · learning rate 0.1 · L1/L2 regularization · SMOTE for class balancing

### 4 — Web Application & Cloud Deployment

Built with **Streamlit** and deployed on **Namal HPC Cloud** — accessible globally via any web browser. Users upload a rice image and receive classified results with bounding boxes, grain measurements, and a downloadable CSV report.

---

## 📊 Results

### Scanner-Trained Model

| Metric | Score |
|--------|:-----:|
| **Overall Accuracy** | **88.86%** |
| Best F1 (Super Brown) | 0.9953 |
| Best F1 (Supri Sella) | 0.9334 |
| Macro Avg F1 | 0.8884 |

### Mobile-Trained Model

| Metric | Score |
|--------|:-----:|
| **Overall Accuracy** | **80.54%** |
| Best F1 (Super Brown) | 0.9941 |
| Best F1 (Supri Sella) | 0.8557 |
| Macro Avg F1 | 0.8048 |

> Super Brown achieved near-perfect classification (F1 > 0.99) across both modes due to its distinctive brown color. Varieties like PK 386 and Super White showed some confusion due to overlapping morphological features.

---

## 🧰 Tech Stack

| Category | Tools & Technologies |
|----------|---------------------|
| **Language** | Python |
| **ML Model** | XGBoost (Gradient Boosting) |
| **Image Processing** | OpenCV, scikit-image |
| **Feature Extraction** | GLCM, LBP, Contour Analysis |
| **Data Handling** | NumPy, Pandas, SMOTE |
| **Web Framework** | Streamlit |
| **Cloud** | Namal HPC Cloud |
| **Hardware (Scanner)** | Canon CanoScan LiDE 300 (600 DPI) |
| **Hardware (Mobile)** | Android smartphone + custom red grid paper |
| **Collaboration** | Alkaram Rice Engineering (PVT) Ltd. |

---

## 🌍 Impact & SDG Alignment

**SDG 9 — Industry, Innovation & Infrastructure:** This project introduces AI-powered automation into Pakistan's agricultural sector, replacing manual inspection with a scalable, cloud-based solution that enhances quality control for rice exports.

**Real-World Impact:**
- Reduces classification time from minutes (manual) to seconds (automated)
- Eliminates human subjectivity in variety identification
- Accessible to both large exporters (scanner mode) and small farmers (mobile mode)
- Deployed and tested with real industry partner data

---

## 📁 Repository Structure

```
├── README.md                          # Project documentation (you are here)
├── NIM-BSEE-2021-19_Final_Report.pdf  # Complete thesis report (60 pages)
├── docs/
│   ├── poster.pdf                     # Project poster
│   ├── user_manual.pdf                # Application user manual
│   └── presentation.pdf               # Final defense slides
├── demo/
│   ├── demo_video.mp4                 # Live demo of the web application
│   └── screenshots/
│       ├── webapp_interface.png        # Main application UI
│       ├── scanner_results.png         # Scanner classification output
│       ├── mobile_results.png          # Mobile classification output
│       ├── confusion_matrix_scanner.png
│       ├── confusion_matrix_mobile.png
│       ├── training_accuracy_scanner.png
│       └── training_accuracy_mobile.png
├── dataset_samples/
│   ├── scanner/                       # Sample scanner images (per variety)
│   └── mobile/                        # Sample mobile images with grid
└── grid_paper/
    └── calibration_grid_template.pdf  # Printable red grid for mobile capture
```

> ⚠️ **Note:** Source code is proprietary to the industry collaboration with Alkaram Rice Engineering and is not publicly shared. This repository showcases the project's methodology, results, and deliverables.

---

## 📄 Documentation

| Document | Description | Link |
|----------|-------------|------|
| 📘 Thesis Report | Complete 60-page FYP report | [View Report](./NIM-BSEE-2021-19_Final_Report.pdf) |
| 🎬 Demo Video | Live walkthrough of the web app | [Watch Demo](./demo/demo_video.mp4) |
| 📊 Poster | Project poster for exhibition | [View Poster](./docs/poster.pdf) |
| 📖 User Manual | Step-by-step usage guide | [Read Manual](./docs/user_manual.pdf) |
| 🎤 Presentation | Final defense slides | [View Slides](./docs/presentation.pdf) |

---

## 📸 Screenshots

<details>
<summary><b>Click to view Web Application & Results</b></summary>

<br>

### Web Application Interface
The Streamlit-based web app provides dual upload options — scanner images (600 DPI recommended) and mobile images (with red grid calibration). Both scanner and mobile models are loaded and ready for real-time classification.

![Web App Interface](./demo/screenshots/webapp_interface.png)

---

### Scanner Classification Results
Uploaded scanner image with detected rice grains highlighted using bounding boxes. Each grain is labeled with its predicted variety. Key measurements (avg length, width, area) and classification breakdown are displayed alongside the image.

![Scanner Results](./demo/screenshots/scanner_results.png)

---

### Mobile Classification Results
Mobile image captured on red grid paper. The system automatically detects the grid for spatial calibration, removes grid lines via inpainting, segments individual grains, and classifies each grain with color-coded labels.

![Mobile Results](./demo/screenshots/mobile_results.png)

---

### Hardware Setup
**Left:** Canon CanoScan LiDE 300 flatbed scanner with rice grains on black background for industrial-grade capture. **Right:** Custom red grid paper (0.25″ squares) used for mobile camera spatial calibration in field conditions.

![Hardware Setup](./demo/screenshots/hardware_setup.png)

---

### Block Diagram
End-to-end system flow — from rice sample placement on scanner/grid paper, through image upload to the cloud web app, to classification results.

![Block Diagram](./demo/screenshots/block_diagram.png)

---

### Training & Validation Accuracy

| Scanner Model | Mobile Model |
|:---:|:---:|
| ![Scanner Accuracy](./demo/screenshots/training_accuracy_scanner.png) | ![Mobile Accuracy](./demo/screenshots/training_accuracy_mobile.png) |

Scanner model reaches ~98% training / ~89% validation accuracy. Mobile model converges at ~85% training / ~80% validation accuracy over 10 epochs.

---

### Confusion Matrices

| Scanner Model (88.86%) | Mobile Model (80.54%) |
|:---:|:---:|
| ![Scanner CM](./demo/screenshots/confusion_matrix_scanner.png) | ![Mobile CM](./demo/screenshots/confusion_matrix_mobile.png) |

Strong diagonal values across both models. Super Brown (1177/1181) and Supri Sella (1101/1181) classified near-perfectly. Minor confusion between PK 386, Super 109, and Super White due to overlapping morphological features.

</details>

---

## 🔮 Future Work

- **Expand dataset** to cover additional rice varieties and broken grain detection
- **Develop a dedicated mobile app** for offline field use
- **Integrate deep learning** (YOLOv8 / EfficientNet) when larger datasets are available
- **Multi-language support** (Urdu, Punjabi) for broader accessibility
- **Mixed-variety purity testing** for export quality assurance

---

## 👥 Team

| Name | Roll Number | Role |
|------|-------------|------|
| **Ahtisham Saleem** | NIM-BSEE-2021-19 | Lead Developer — ML Pipeline, Feature Engineering, Web App, Cloud Deployment |
| **Muhammad Yousaf** | NIM-BSEE-2021-30 | Dataset Collection, Image Processing, Testing,Cloud Deployment |

---

## 🏫 Academic Info

| Detail | Info |
|--------|------|
| **Project Type** | Final Year Project (FYP) — BS Electrical Engineering |
| **University** | Namal University, Mianwali |
| **Supervisor** | Dr. Tassadaq Hussain |
| **Co-Supervisor** | Dr. Farukh Qureshi |
| **Industry Partner** | Alkaram Rice Engineering (PVT) Ltd. |
| **Research Lab** | Haji Muhammad Rice Mill (Dataset Collection) |
| **Supporting Centre** | Centre for AI & Big Data, Namal University |
| **Year** | 2024–2025 |
| **Turnitin Score** | 5% (Original Work) |

---

## 📬 Contact

- **Email:** engr.ahtishamsaleem@gmail.com
- **LinkedIn:** [Ahtisham Saleem](https://www.linkedin.com/in/ahtisham-salim)
- **GitHub:** [@codebyahtisham](https://github.com/codebyahtisham)

---

<p align="center">
  ⭐ If you found this project interesting or useful, consider giving it a star!
</p>
