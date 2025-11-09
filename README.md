# Realistic Image Capture of Surface Defects

This repository contains Python scripts and data used to evaluate surface defect visibility under different lighting angles and wavelengths.  
The work was conducted as part of the **PAI700 course at University West (2025)**.

---

## 📘 Overview
The project investigates how lighting geometry and wavelength affect defect visibility across different materials — **metal, plastic, and transparent surfaces**.  
Two Python scripts were developed to preprocess images, compute visibility metrics, and determine optimal illumination conditions.

---

## ⚙️ Files Included
- `image_processing_pipeline.py` – Performs image preprocessing, normalization, and glare suppression.  
- `compute_image_quality_metrics.py` – Calculates CNR, SNR, sharpness, entropy, and glare ratio.  
- `manifest.csv`, `metrics.csv`, `ranking_summary.csv` – Contain computed results for each test condition.  

---

## 🧮 Key Findings
- **45° Red Light** – Optimal for opaque materials (metal, plastic).  
- **0° Green Light** – Best for transparent materials.  
- Illumination geometry had a stronger impact on defect visibility than wavelength.  

---

## 🧰 Requirements
- Python 3.10 or higher  
- Libraries: `opencv-python`, `numpy`, `pandas`, `matplotlib`  

Install dependencies using:
```bash
pip install opencv-python numpy pandas matplotlib
