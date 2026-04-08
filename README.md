# NeuroLens — Brain Tumor Classification via Topology-Fused Tensor Learning

A research-grade web application for explainable brain tumor classification from MRI scans. It fuses **Topological Data Analysis (TDA)**, **Tucker/Tensor Train decomposition**, and classical ML classifiers into a novel pipeline — achieving **92.45% accuracy** with full GradCAM++ visualizations and ROI explainability.

---

## What It Does

Upload an MRI scan, select a classifier, and get:

- **4-class prediction** — Glioma, Meningioma, No Tumor, Pituitary
- **GradCAM++ heatmap** — CNN attention overlay on the scan
- **TDA ROI mask** — tumor region extracted via persistent homology
- **Explainability metrics** — IoU and Dice score alignment between GradCAM++ and TDA ROI
- **Analysis history** — browsable report log with thumbnails

---

## Novel Technical Contributions

| Component | Description |
|-----------|-------------|
| TDA-Weighted Tucker Decomposition | Persistent homology ROI masks as attention weights during Tucker feature extraction |
| Hybrid Tensor Fusion | Tucker (256 dims) + Tensor Train (144 dims) + TDA (up to 840 dims) → PCA-fused ~407 dims |
| Multi-Scale Persistent Homology | H0 (connected components) + H1 (loops) at scales [10, 20] |
| Topological Regularization Loss | CNN training penalized by persistence-aware topological loss (λ=0.05) |
| GradCAM++ Explainer | Second-order gradient weighting for faithful heatmaps on a 4-block CNN |
| Quantitative Explainability | IoU/Dice alignment between GradCAM++ and TDA ROI as interpretability scores |

---

## Model Performance

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| **SVM (RBF)** | **92.45%** | **92.38%** |
| LightGBM | 90.16% | — |
| KNN | 89.32% | — |
| XGBoost | 89.02% | — |
| Random Forest | 85.05% | — |
| Extra Trees | 83.98% | — |

All models are trained on the same fused feature vector; the CNN is used only as a GradCAM++ explainer, not the primary classifier.

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Web | Flask 3.0+, Waitress (WSGI), Jinja2 |
| ML | scikit-learn, XGBoost, LightGBM |
| Deep Learning | PyTorch 2.2+ (SimpleBrainCNN, GradCAM++) |
| Topology | ripser (persistent homology), scipy |
| Tensors | TensorLy (Tucker, Tensor Train) |
| Vision | OpenCV, Pillow, NumPy |
| Frontend | HTML5, CSS3, JavaScript (ES6+) |

---

## Dataset

[Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- 5,712 training images, 1,311 test images
- 250×250 grayscale MRI scans

Place it at:
```
data/Dataset/Training/{glioma,meningioma,notumor,pituitary}/
data/Dataset/Testing/{glioma,meningioma,notumor,pituitary}/
```

---

## Installation

### Option A — Conda (recommended)

```bash
conda create -n neurolens python=3.11 -y
conda activate neurolens
pip install -r requirements.txt
```

### Option B — venv (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option C — venv (Windows)

```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the App

### First-time build (choose TDA complexity)

```bash
# Fast — ~3 min | H0 only, 110 TDA dims, 6 CNN epochs | good for demos
python app.py --prepare --tda-mode fast

# Standard — ~10 min | H0+H1, 440 TDA dims, 8 CNN epochs | default
python app.py --prepare --tda-mode standard

# Full — ~25 min | H0+H1 + persistence images, 840 TDA dims, 12 CNN epochs | best results
python app.py --prepare --tda-mode full
```

Build artifacts are cached at `output/webapp_artifacts/`. Subsequent starts are near-instant.

### Start the server

```bash
# Development
python app.py

# Production (Waitress WSGI)
python serve.py
```

Open `http://localhost:8000` (auto-falls back to `:8001` if port is busy).

### Pages

| URL | Description |
|-----|-------------|
| `/` | Home — overview and latest analysis |
| `/analyze` | Upload MRI, select model, get prediction |
| `/history` | Last 12 analyses with thumbnails |
| `/api/predict` | POST endpoint — file upload + model selection |
| `/api/health` | Service status |
| `/api/models` | List available classifiers |

---

## Project Structure

```
.
├── app.py                        # Flask entry point
├── serve.py                      # Waitress production server
├── wsgi.py                       # WSGI application
├── requirements.txt
├── brain_tumor_web/
│   ├── inference.py              # Core pipeline (TDA, tensors, classifiers, GradCAM++)
│   ├── static/
│   │   ├── app.js                # Frontend logic (drag-drop, upload, API calls)
│   │   └── styles.css            # UI styling
│   └── templates/                # Jinja2 HTML templates
│       ├── base.html
│       ├── home.html
│       ├── analyze.html
│       ├── history.html
│       └── report.html
├── data/Dataset/                 # MRI dataset (not tracked in git)
├── output/
│   ├── models/                   # Trained classifier joblib files
│   ├── webapp_artifacts/         # Runtime bundle (CNN + classical pipeline)
│   ├── webapp_data/reports/      # Per-analysis JSON + images
│   └── metrics/                  # Cached TDA features, evaluation JSON
└── notebooke30d8b8ae0-2-1.ipynb  # Training and analysis notebook
```

---

## Key Configuration (inference.py)

```python
IMG_SIZE             = 250          # MRI input resolution
CNN_IMAGE_SIZE       = 128          # CNN input size
TDA_SCALES           = [10, 20]     # Multi-scale smoothing for topology
FAST_TUCKER_RANK     = (16, 16, 64) # Tucker decomposition rank
TT_RANK              = 12           # Tensor Train rank
FUSION_PCA_VARIANCE  = 0.99         # PCA variance retention after fusion
TOPO_REG_WEIGHT      = 0.05         # Topological regularization strength (λ)
GRADCAM_SMOOTH_SIGMA = 1.5          # Gaussian smoothing for CAM heatmap
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
