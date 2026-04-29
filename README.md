# Brain Tumor Classification System

A multimodal deep learning system for brain tumor classification and analysis, featuring a React frontend and a Flask/Python backend.

## Project Structure

- **BrainTumorClass/**: Backend logic, models, and API.
  - `api.py`: Flask API for model serving.
  - `predict.py`: Core prediction and inference logic.
  - `src/`: Source modules for data loading, classification, and XAI (Grad-CAM).
- **frontend/**: React + Vite application for the user interface.
  - `src/`: UI components, pages, and API services.

## Setup

### Backend
1. Navigate to `BrainTumorClass/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the API: `python api.py`.

### Frontend
1. Navigate to `frontend/`.
2. Install dependencies: `npm install`.
3. Run the dev server: `npm run dev`.

## Features
- Real-time brain tumor classification.
- Probability distribution visualization.
- XAI (Grad-CAM) for model interpretability.
- Diagnostic plotting and pipeline visualization.
