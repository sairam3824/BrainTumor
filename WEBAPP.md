# Brain Tumor Web App

## What it does

- Drag-and-drop or browse to upload an MRI image
- Choose a classifier, with the notebook best model as the default
- Run the topology-fused classifier pipeline for prediction
- Show a Grad-CAM overlay from the explainer CNN

## Run it

```bash
pip install -r requirements.txt
python app.py
```

Then open [http://localhost:8000](http://localhost:8000).

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
