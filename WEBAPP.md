# Brain Tumor Web App

## What it does

- Home page with overview and quick links
- Analyze page with drag-and-drop upload, model selection, progress tracking, and prediction
- History page for previously completed analyses
- Grad-CAM overlay from the explainer CNN

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:8000](http://localhost:8000).

If `8000` is already occupied, the app automatically falls back to `http://localhost:8001`.

## One-time preparation

On the first run, the app builds cached assets under `output/webapp_artifacts/`.
That includes:

- a rebuilt fused-classifier inference bundle
- a CNN checkpoint used for Grad-CAM

If you want to prepare those assets before opening the UI:

```bash
python app.py --prepare
```

## Notes

- The default classifier is the notebook best model: `SVM`
- The Grad-CAM heatmap comes from the app's CNN explainer, not from the classical model itself
- The selectable models exposed immediately are the stable sklearn-backed set: `Extra Trees`, `Random Forest`, `SVM`, and `KNN`
- The default port is `8000` because `5000` is commonly occupied by other macOS services

## Production-style local serve

For a cleaner non-dev launch path:

```bash
python serve.py
```

That uses `waitress` on `http://localhost:8000`, and automatically falls back to `http://localhost:8001` if needed.
