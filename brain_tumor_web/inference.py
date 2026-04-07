from __future__ import annotations

import io
import json
import math
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from ripser import lower_star_img, ripser
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import __version__ as sklearn_version
from tensorly.decomposition import tucker


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "brain_tumor_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ALLOWED_MIME_PREFIXES = ("image/",)
MAX_IMAGE_PIXELS = 20_000_000
IMG_SIZE = 250
CNN_IMAGE_SIZE = 128
GAUSSIAN_BLUR_KERNEL = (5, 5)
CONTRAST_ALPHA = 1.5
CONTRAST_BETA = 80
TDA_SCALES = [10, 20]
TDA_BORDER_WIDTHS = [35, 70]
TDA_MAX_HOMOLOGY_DIM = 1
BETTI_CURVE_RESOLUTION = 100
FAST_TDA_H1_MAX_POINTS = 180
FAST_TUCKER_RANK = (16, 16, 64)
FAST_TUCKER_ALPHA = 1.5
FAST_TUCKER_BG_WEIGHT = 0.7
FAST_TUCKER_MAX_TRAIN_SAMPLES = 1200
FUSION_PCA_VARIANCE = 0.99
RANDOM_STATE = 42
BATCH_SIZE = 64
CNN_EPOCHS = 4
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 1e-3
HISTORY_LIMIT = 100
STATUS_PROGRESS = {
    "idle": 0,
    "starting": 8,
    "building": 48,
    "ready": 100,
    "error": 100,
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label_index: int


class SimpleBrainCNN(nn.Module):
    def __init__(self, num_classes: int = len(LABELS), embed_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(128, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.block4[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        feature_maps = self.block4(x)
        pooled = self.pool(feature_maps).flatten(1)
        embedding = F.relu(self.embedding(pooled))
        return self.classifier(self.dropout(embedding))


class BrainTumorDataset(torch.utils.data.Dataset):
    def __init__(self, records: Sequence[ImageRecord], image_size: int) -> None:
        self.records = list(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = load_image_file(record.path, image_size=self.image_size)
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(record.label_index, dtype=torch.long)
        return tensor, label


class BrainTumorWebService:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.dataset_base = self.base_dir / "data" / "Dataset"
        self.output_dir = self.base_dir / "output"
        self.metrics_path = self.output_dir / "metrics" / "full_results.json"
        self.tda_cache_path = self.output_dir / "metrics" / "tda_train.npy"
        self.roi_cache_path = self.output_dir / "metrics" / "roi_train.npy"
        self.artifact_dir = self.output_dir / "webapp_artifacts"
        self.bundle_path = self.artifact_dir / "classical_bundle.joblib"
        self.cnn_path = self.artifact_dir / "gradcam_cnn.pth"
        self.manifest_path = self.artifact_dir / "runtime_manifest.json"
        self.media_root = self.output_dir / "webapp_data"
        self.reports_dir = self.media_root / "reports"
        self.history_path = self.media_root / "history.json"
        self.status = "idle"
        self.message = "The analysis runtime is idle."
        self.progress = 0
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._bundle = None
        self._cnn_model: SimpleBrainCNN | None = None
        self._metadata = self._load_metadata()

        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.media_root.mkdir(parents=True, exist_ok=True)
        if not self.history_path.exists():
            write_json_atomic(self.history_path, [])

        self._attempt_load_cached_assets()

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def _set_status(self, status: str, message: str, progress: int | None = None, *, error: str | None = None) -> None:
        self.status = status
        self.message = message
        self.progress = STATUS_PROGRESS.get(status, 0) if progress is None else progress
        self.error = error

    def _attempt_load_cached_assets(self) -> None:
        try:
            if self._manifest_is_compatible() and self.bundle_path.exists():
                self._bundle = joblib.load(self.bundle_path)
            if self._manifest_is_compatible() and self.cnn_path.exists():
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = SimpleBrainCNN().to(device)
                checkpoint = torch.load(self.cnn_path, map_location=device)
                model.load_state_dict(checkpoint["state_dict"])
                model.eval()
                self._cnn_model = model
            if self._bundle is not None and self._cnn_model is not None:
                self._set_status("ready", "Prebuilt runtime assets loaded successfully.", 100)
        except Exception as exc:
            self._bundle = None
            self._cnn_model = None
            self._set_status("idle", "Prebuilt assets were unavailable, so the app will rebuild them on demand.", 0, error=str(exc))

    def _manifest_is_compatible(self) -> bool:
        if not self.manifest_path.exists():
            return False
        try:
            with self.manifest_path.open() as handle:
                manifest = json.load(handle)
        except Exception:
            return False
        return manifest.get("app_format") == 2 and manifest.get("sklearn_version") == sklearn_version

    def health_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "assets_ready": self.is_ready,
            "report_count": len(self.list_history(limit=HISTORY_LIMIT)),
        }
        if self.error:
            payload["error"] = self.error
        return payload

    def models_payload(self) -> Dict[str, object]:
        metadata_models = self._metadata.get("all_model_results", {})
        model_names = list(self._bundle["model_order"]) if self._bundle is not None else ["Extra Trees", "Random Forest", "SVM", "KNN"]
        default_model = self._resolve_default_model(model_names)

        models = []
        for name in model_names:
            notebook_accuracy = metadata_models.get(name, {}).get("avg_metrics", {}).get("Accuracy")
            models.append(
                {
                    "name": name,
                    "label": f"{name} (Default)" if name == default_model else name,
                    "notebook_accuracy": f"{notebook_accuracy:.4f}" if isinstance(notebook_accuracy, (int, float)) else "Unavailable",
                }
            )

        default_accuracy = metadata_models.get(default_model, {}).get("avg_metrics", {}).get("Accuracy")
        return {
            "models": models,
            "default_model": default_model,
            "default_model_label": default_model or "Unavailable",
            "default_model_accuracy": f"{default_accuracy:.4f}" if isinstance(default_accuracy, (int, float)) else "Unavailable",
            "default_accuracy_label": "Notebook reference accuracy",
        }

    def _resolve_default_model(self, model_names: Sequence[str]) -> str:
        preferred = self._metadata.get("best_model")
        if preferred in model_names:
            return preferred
        return model_names[0] if model_names else "Unavailable"

    def ensure_started(self) -> None:
        with self._lock:
            if self.status in {"starting", "building", "ready"}:
                return
            self._set_status("starting", "Booting the background asset builder.", 8)
            self._thread = threading.Thread(target=self._prepare_assets, daemon=True)
            self._thread.start()

    def prepare_blocking(self) -> None:
        self._prepare_assets()

    def _prepare_assets(self) -> None:
        try:
            self._set_status("building", "Loading or rebuilding the fused inference bundle.", 18)
            self._bundle = self._load_or_build_bundle()

            self._set_status("building", "Loading or training the Grad-CAM explainer.", 72)
            self._cnn_model = self._load_or_train_cnn()

            self._set_status("ready", "Models are ready. Upload an MRI slice to begin analysis.", 100)
        except Exception as exc:
            self._set_status("error", f"Asset preparation failed: {exc}", 100, error=str(exc))

    def _load_or_build_bundle(self):
        if self.bundle_path.exists():
            self._set_status("building", "Loading prebuilt classifier assets from disk.", 32)
            return joblib.load(self.bundle_path)

        records = collect_image_records(self.dataset_base / "Training")
        if not records:
            raise RuntimeError("No training images were found in data/Dataset/Training.")
        if not self.tda_cache_path.exists() or not self.roi_cache_path.exists():
            raise RuntimeError("Missing output/metrics/tda_train.npy or output/metrics/roi_train.npy.")

        self._set_status("building", "Scanning training images and preparing topology features.", 40)
        tda_train = np.load(self.tda_cache_path)
        roi_train = np.load(self.roi_cache_path, mmap_mode="r")
        if len(records) != len(tda_train) or len(records) != len(roi_train):
            raise RuntimeError("Cached TDA/ROI arrays do not match the training split size.")

        self._set_status("building", "Projecting weighted Tucker features for the live bundle.", 52)
        a_factor, b_factor = fit_weighted_tucker_factors(records, roi_train)
        train_tucker = build_train_tucker_features(records, roi_train, a_factor, b_factor)
        fusion_pipeline = fit_fusion_pipeline(train_tucker, tda_train)
        fused_train = transform_fused_features(train_tucker, tda_train, fusion_pipeline)
        y_train = np.asarray([record.label_index for record in records], dtype=np.int64)

        self._set_status("building", "Training the stable classifier set for interactive analysis.", 64)
        trained_models = train_classifiers(fused_train, y_train)
        model_order = list(trained_models.keys())
        best_model = self._resolve_default_model(model_order)

        bundle = {
            "best_model": best_model,
            "labels": LABELS,
            "A": a_factor.astype(np.float32),
            "B": b_factor.astype(np.float32),
            "fusion_pipeline": fusion_pipeline,
            "models": trained_models,
            "model_order": model_order,
            "built_at": utcnow_iso(),
        }
        joblib.dump(bundle, self.bundle_path)
        write_json_atomic(
            self.manifest_path,
            {
                "app_format": 2,
                "sklearn_version": sklearn_version,
                "built_at": bundle["built_at"],
            },
        )
        return bundle

    def _load_or_train_cnn(self) -> SimpleBrainCNN:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleBrainCNN().to(device)
        if self.cnn_path.exists():
            checkpoint = torch.load(self.cnn_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            return model

        records = collect_image_records(self.dataset_base / "Training")
        if not records:
            raise RuntimeError("No training images were found for the Grad-CAM explainer.")

        train_cnn_model(model, records, device=device)
        torch.save({"state_dict": model.state_dict()}, self.cnn_path)
        model.eval()
        return model

    def validate_upload(self, file_bytes: bytes, filename: str | None, mimetype: str | None) -> np.ndarray:
        name = (filename or "").strip()
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
            raise ValueError(f"Unsupported file type. Use one of: {allowed}.")

        if mimetype and not any(mimetype.startswith(prefix) for prefix in ALLOWED_MIME_PREFIXES):
            raise ValueError("The uploaded file is not recognized as an image.")

        try:
            with Image.open(io.BytesIO(file_bytes)) as image:
                image.load()
                width, height = image.size
                if width < 32 or height < 32:
                    raise ValueError("Please upload an image that is at least 32x32 pixels.")
                if width * height > MAX_IMAGE_PIXELS:
                    raise ValueError("The uploaded image is too large to analyze safely.")
                return np.asarray(image.convert("RGB"))
        except UnidentifiedImageError as exc:
            raise ValueError("The uploaded file could not be decoded as an image.") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read the uploaded image: {exc}") from exc

    def predict(self, file_bytes: bytes, *, filename: str | None, mimetype: str | None, model_name: str | None = None) -> Dict[str, object]:
        if self._bundle is None:
            raise ValueError("The inference bundle is not ready yet.")

        selected_model = model_name or self._bundle["best_model"]
        if selected_model not in self._bundle["models"]:
            raise ValueError(f"Unknown model '{selected_model}'.")

        source_image = self.validate_upload(file_bytes, filename, mimetype)
        inference_image = preprocess_uploaded_array(source_image, image_size=IMG_SIZE)
        tda_feature = extract_compact_tda_feature(inference_image)
        tucker_feature = project_images(np.expand_dims(inference_image, axis=0), self._bundle["A"], self._bundle["B"])[0]
        fused = transform_fused_features(
            tucker_feature[np.newaxis, :],
            tda_feature[np.newaxis, :],
            self._bundle["fusion_pipeline"],
        )

        estimator = self._bundle["models"][selected_model]
        probs = score_estimator(estimator, fused)[0]
        pred_idx = int(np.argmax(probs))
        sorted_scores = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)
        scores = [
            {
                "label": LABELS[idx],
                "value": float(prob),
                "percent": float(prob * 100.0),
            }
            for idx, prob in sorted_scores
        ]

        input_url = None
        gradcam_url = None
        gradcam_note = "Grad-CAM overlay generated by the explainer CNN."
        if self._cnn_model is not None:
            cam = generate_gradcam(
                self._cnn_model,
                preprocess_uploaded_array(source_image, image_size=CNN_IMAGE_SIZE),
                target_class=pred_idx,
            )
            overlay = make_heatmap_overlay(source_image, cam)
        else:
            overlay = None
            gradcam_note = "Grad-CAM explainer was unavailable for this request."

        analysis_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        created_at = utcnow_iso()

        with self._lock:
            input_path = self.reports_dir / f"{analysis_id}-input.png"
            save_rgb_image(source_image, input_path)
            input_url = self.asset_url(input_path.relative_to(self.media_root))

            if overlay is not None:
                gradcam_path = self.reports_dir / f"{analysis_id}-gradcam.png"
                save_rgb_image(overlay, gradcam_path)
                gradcam_url = self.asset_url(gradcam_path.relative_to(self.media_root))

            notebook_accuracy = (
                self._metadata.get("all_model_results", {})
                .get(selected_model, {})
                .get("avg_metrics", {})
                .get("Accuracy")
            )
            report = {
                "analysis_id": analysis_id,
                "created_at": created_at,
                "predicted_label": LABELS[pred_idx],
                "predicted_index": pred_idx,
                "top_confidence_percent": round(float(sorted_scores[0][1] * 100.0), 2),
                "selected_model": selected_model,
                "default_model": self._bundle["best_model"],
                "prediction_note": f"Classifier output from {selected_model}. Grad-CAM is produced by the explainer CNN.",
                "scores": scores,
                "input_image_url": input_url,
                "gradcam_image_url": gradcam_url,
                "gradcam_note": gradcam_note,
                "notebook_reference_accuracy": round(float(notebook_accuracy), 4) if isinstance(notebook_accuracy, (int, float)) else None,
                "notebook_reference_label": "Notebook reference accuracy",
            }
            self._persist_report(report)

        return {
            **report,
            "report_url": f"/report/{analysis_id}",
            "history_url": "/history",
            "model_label": selected_model,
        }

    def _persist_report(self, report: Dict[str, object]) -> None:
        report_path = self.reports_dir / f"{report['analysis_id']}.json"
        write_json_atomic(report_path, report)

        history = self._load_history()
        history.insert(
            0,
            {
                "analysis_id": report["analysis_id"],
                "created_at": report["created_at"],
                "predicted_label": report["predicted_label"],
                "selected_model": report["selected_model"],
                "top_confidence_percent": report["top_confidence_percent"],
                "report_url": f"/report/{report['analysis_id']}",
                "input_image_url": report["input_image_url"],
                "gradcam_image_url": report["gradcam_image_url"],
            },
        )
        history = history[:HISTORY_LIMIT]
        write_json_atomic(self.history_path, history)

    def _load_metadata(self) -> Dict[str, object]:
        if not self.metrics_path.exists():
            return {}
        with self.metrics_path.open() as handle:
            return json.load(handle)

    def _load_history(self) -> List[Dict[str, object]]:
        if not self.history_path.exists():
            return []
        with self.history_path.open() as handle:
            return json.load(handle)

    def list_history(self, limit: int = 20) -> List[Dict[str, object]]:
        history = self._load_history()
        return history[:limit]

    def get_report(self, analysis_id: str | None = None) -> Dict[str, object] | None:
        target = analysis_id
        if target is None:
            history = self.list_history(limit=1)
            if not history:
                return None
            target = history[0]["analysis_id"]

        report_path = self.reports_dir / f"{target}.json"
        if not report_path.exists():
            return None
        with report_path.open() as handle:
            return json.load(handle)

    def asset_url(self, relative_path: Path) -> str:
        return f"/artifacts/{relative_path.as_posix()}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def collect_image_records(split_dir: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for label_index, label in enumerate(LABELS):
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file():
                records.append(ImageRecord(path=path, label_index=label_index))
    return records


def load_image_file(path: Path, image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return preprocess_grayscale(image, image_size=image_size)


def preprocess_uploaded_array(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    return preprocess_grayscale(gray, image_size=image_size)


def preprocess_grayscale(image: np.ndarray, image_size: int) -> np.ndarray:
    image = cv2.GaussianBlur(image, GAUSSIAN_BLUR_KERNEL, 0)
    image = cv2.convertScaleAbs(image, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)
    image = cv2.resize(image, (image_size, image_size))
    image = 255 - image
    return image.astype(np.float32) / 255.0


def load_batch_images(records: Sequence[ImageRecord], image_size: int = IMG_SIZE) -> np.ndarray:
    images = [load_image_file(record.path, image_size=image_size) for record in records]
    return np.stack(images).astype(np.float32)


def smoothen(img: np.ndarray, window_size: int) -> np.ndarray:
    return ndimage.uniform_filter(img.astype(float), size=window_size)


def add_border(img: np.ndarray, border_width: int) -> np.ndarray:
    bordered = img.copy()
    border_value = float(np.min(img) - 1)
    bordered[0:border_width, :] = border_value
    bordered[(bordered.shape[0] - border_width): bordered.shape[0], :] = border_value
    bordered[:, 0:border_width] = border_value
    bordered[:, (bordered.shape[1] - border_width): bordered.shape[1]] = border_value
    return bordered


def img_to_sparse_dm(img: np.ndarray):
    m, n = img.shape
    idxs = np.arange(m * n).reshape((m, n))
    i_vals = idxs.flatten()
    j_vals = idxs.flatten()
    values = img.flatten()

    tiled_indices = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tiled_indices[1:-1, 1:-1] = idxs
    tiled_distances = np.ones_like(tiled_indices) * np.nan
    tiled_distances[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            shifted_indices = np.roll(np.roll(tiled_indices, di, axis=0), dj, axis=1)
            shifted_distances = np.roll(np.roll(tiled_distances, di, axis=0), dj, axis=1)
            shifted_distances = np.maximum(shifted_distances, tiled_distances)
            boundary = ~np.isnan(shifted_distances)
            i_vals = np.concatenate((i_vals, tiled_indices[boundary].flatten()))
            j_vals = np.concatenate((j_vals, shifted_indices[boundary].flatten()))
            values = np.concatenate((values, shifted_distances[boundary].flatten()))

    return sparse.coo_matrix((values, (i_vals, j_vals)), shape=(idxs.size, idxs.size))


def connected_components_img(img: np.ndarray) -> np.ndarray:
    m, n = img.shape
    return connected_components(img_to_sparse_dm(img), directed=False)[1].reshape((m, n))


def lifetimes_from_dgm(dgm: np.ndarray, compute_tau: bool = False):
    dgm_lifetimes = np.vstack([dgm[:, 0], dgm[:, 1] - dgm[:, 0]]).T
    if not compute_tau:
        return dgm_lifetimes

    finite = np.delete(dgm_lifetimes.copy(), np.where(dgm_lifetimes[:, 1] == np.inf)[0], axis=0)
    sorted_points = finite[:, 1].copy()
    sorted_points[::-1].sort()
    dist_to_next = np.delete(sorted_points, len(sorted_points) - 1) - np.delete(sorted_points, 0)
    most_distant = np.argmax(dist_to_next)
    tau = (sorted_points[most_distant] + sorted_points[most_distant + 1]) / 2
    return dgm_lifetimes, tau


def topological_process_img(img: np.ndarray, window_size: int = 10, border_width: int = 70) -> np.ndarray:
    processed = smoothen(img.copy(), window_size=window_size)
    processed = add_border(processed, border_width=border_width)

    dgm = lower_star_img(processed)
    dgm_lifetimes, tau = lifetimes_from_dgm(dgm, compute_tau=True)
    idxs = np.where(np.logical_and(tau < dgm_lifetimes[:, 1], dgm_lifetimes[:, 1] < np.inf))[0]
    idxs = np.flip(idxs[np.argsort(dgm[idxs, 0])])
    death_indices = np.zeros(0).astype(int)
    img_components = np.zeros_like(processed)

    for idx in idxs:
        birth_index = np.argmin(np.abs(processed - dgm[idx, 0]))
        death_indices = np.append(death_indices, np.argmin(np.abs(processed - dgm[idx, 1])))

        img_temp = np.ones_like(processed)
        img_temp[np.logical_or(processed < dgm[idx, 0] - 0.01, dgm[idx, 1] - 0.01 < processed)] = np.nan
        component = connected_components_img(img_temp)
        component = component == component[birth_index // processed.shape[1], birth_index % processed.shape[1]]
        if len(death_indices) > 1:
            in_component = idxs[
                np.where([component[d // processed.shape[1], d % processed.shape[1]] for d in death_indices])[0]
            ]
            if len(in_component) > 0:
                component[processed > np.min(dgm[in_component, 1]) - 0.1] = False
        img_components[component] = 1

    return img_components.astype(np.float32)


def persistence_statistics(dgm: np.ndarray) -> np.ndarray:
    if len(dgm) == 0:
        return np.zeros(10, dtype=np.float32)

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    lifetimes = deaths - births
    positive = lifetimes[lifetimes > 0]
    if len(positive) > 0:
        total = np.sum(positive)
        probs = positive / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
    else:
        entropy = 0.0

    return np.array(
        [
            len(dgm),
            np.mean(births),
            np.std(births),
            np.mean(deaths),
            np.std(deaths),
            np.mean(lifetimes),
            np.std(lifetimes),
            np.max(lifetimes),
            np.sum(lifetimes),
            entropy,
        ],
        dtype=np.float32,
    )


def betti_curve(dgm: np.ndarray, resolution: int = BETTI_CURVE_RESOLUTION) -> np.ndarray:
    if len(dgm) == 0:
        return np.zeros(resolution, dtype=np.float32)

    births = dgm[:, 0]
    deaths = dgm[:, 1]
    t_grid = np.linspace(np.min(births), np.max(deaths), resolution)
    curve = np.zeros(resolution, dtype=np.float32)
    for idx, point in enumerate(t_grid):
        curve[idx] = np.sum((births <= point) & (deaths > point))
    return curve


def compute_h1_persistence(img: np.ndarray, window_size: int = 10, max_points: int = FAST_TDA_H1_MAX_POINTS) -> np.ndarray:
    img_smooth = smoothen(img.copy(), window_size=window_size)
    height, width = img_smooth.shape
    threshold = np.percentile(img_smooth, 20)

    step = max(1, int(math.sqrt((height * width) / max_points)))
    rr = np.arange(0, height, step)
    cc = np.arange(0, width, step)
    r_grid, c_grid = np.meshgrid(rr, cc, indexing="ij")
    vals = img_smooth[r_grid, c_grid]
    keep = vals > threshold

    if int(keep.sum()) < 10:
        return np.empty((0, 2), dtype=np.float32)

    coords = np.column_stack([r_grid[keep], c_grid[keep]]).astype(np.float64)
    vals = vals[keep].astype(np.float64)
    coords_max = np.maximum(coords.max(axis=0, keepdims=True), 1.0)
    coords_norm = coords / coords_max
    vals_norm = vals / (vals.max() + 1e-8)
    point_cloud = np.column_stack([coords_norm, vals_norm * 0.5])

    if len(point_cloud) > max_points:
        idx = np.linspace(0, len(point_cloud) - 1, max_points, dtype=int)
        point_cloud = point_cloud[idx]

    result = ripser(point_cloud, maxdim=1, thresh=2.0)
    dgm_h1 = result["dgms"][1]
    if len(dgm_h1) == 0:
        return np.empty((0, 2), dtype=np.float32)
    dgm_h1 = dgm_h1[np.isfinite(dgm_h1[:, 1])]
    return dgm_h1.astype(np.float32)


def compute_multiscale_persistence(img: np.ndarray, scales: Sequence[int] = TDA_SCALES) -> Dict[str, np.ndarray]:
    diagrams: Dict[str, np.ndarray] = {}
    for scale in scales:
        border_width = TDA_BORDER_WIDTHS[0] if scale <= 10 else TDA_BORDER_WIDTHS[-1]
        img_smooth = smoothen(img.copy(), window_size=scale)
        dgm_h0 = lower_star_img(add_border(img_smooth, border_width=border_width))
        dgm_h0 = dgm_h0[np.isfinite(dgm_h0[:, 1])]
        diagrams[f"h0_scale{scale}"] = dgm_h0.astype(np.float32)
        if TDA_MAX_HOMOLOGY_DIM >= 1:
            diagrams[f"h1_scale{scale}"] = compute_h1_persistence(img, window_size=scale)
    return diagrams


def compact_feature_from_diagrams(diagrams: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for _, dgm in diagrams.items():
        parts.append(persistence_statistics(dgm))
        parts.append(betti_curve(dgm))
    return np.concatenate(parts).astype(np.float32)


def extract_compact_tda_feature(img: np.ndarray) -> np.ndarray:
    return compact_feature_from_diagrams(compute_multiscale_persistence(img))


def fit_weighted_tucker_factors(records: Sequence[ImageRecord], roi_train: np.ndarray):
    rng = np.random.default_rng(RANDOM_STATE)
    subset_size = min(FAST_TUCKER_MAX_TRAIN_SAMPLES, len(records))
    subset_idx = np.sort(rng.choice(len(records), size=subset_size, replace=False))
    subset_records = [records[idx] for idx in subset_idx]
    subset_images = load_batch_images(subset_records, image_size=IMG_SIZE)
    subset_roi = np.asarray(roi_train[subset_idx], dtype=np.float32)
    weighted_subset = subset_images * (FAST_TUCKER_BG_WEIGHT + FAST_TUCKER_ALPHA * subset_roi)
    tensor_subset = np.transpose(weighted_subset, (1, 2, 0))
    _, factors = tucker(
        tensor_subset,
        rank=(FAST_TUCKER_RANK[0], FAST_TUCKER_RANK[1], min(FAST_TUCKER_RANK[2], subset_size)),
    )
    a_factor, b_factor, _ = factors
    return a_factor.astype(np.float32), b_factor.astype(np.float32)


def project_images(images: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> np.ndarray:
    projected = np.einsum("bhw,hr,ws->brs", images, a_factor, b_factor)
    return projected.reshape(projected.shape[0], -1).astype(np.float32)


def build_train_tucker_features(records: Sequence[ImageRecord], roi_train: np.ndarray, a_factor: np.ndarray, b_factor: np.ndarray) -> np.ndarray:
    features = []
    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start:start + BATCH_SIZE]
        batch_images = load_batch_images(batch_records, image_size=IMG_SIZE)
        roi_batch = np.asarray(roi_train[start:start + len(batch_records)], dtype=np.float32)
        weighted_batch = batch_images * (FAST_TUCKER_BG_WEIGHT + FAST_TUCKER_ALPHA * roi_batch)
        features.append(project_images(weighted_batch, a_factor, b_factor))
    return np.vstack(features).astype(np.float32)


def fit_fusion_pipeline(tucker_features_train: np.ndarray, tda_features_train: np.ndarray):
    scaler_tucker = StandardScaler()
    scaler_tda = StandardScaler()
    tucker_train_norm = scaler_tucker.fit_transform(np.nan_to_num(tucker_features_train, nan=0.0, posinf=0.0, neginf=0.0))
    tda_train_norm = scaler_tda.fit_transform(np.nan_to_num(tda_features_train, nan=0.0, posinf=0.0, neginf=0.0))
    concat = np.hstack([tucker_train_norm, tda_train_norm])
    pca = PCA(n_components=FUSION_PCA_VARIANCE, svd_solver="full")
    pca.fit(concat)
    return {
        "scaler_tucker": scaler_tucker,
        "scaler_tda": scaler_tda,
        "pca": pca,
    }


def transform_fused_features(tucker_features: np.ndarray, tda_features: np.ndarray, fusion_pipeline: Dict[str, object]) -> np.ndarray:
    tucker = np.nan_to_num(tucker_features, nan=0.0, posinf=0.0, neginf=0.0)
    tda = np.nan_to_num(tda_features, nan=0.0, posinf=0.0, neginf=0.0)
    tucker_scaled = fusion_pipeline["scaler_tucker"].transform(tucker)
    tda_scaled = fusion_pipeline["scaler_tda"].transform(tda)
    return fusion_pipeline["pca"].transform(np.hstack([tucker_scaled, tda_scaled]))


def train_classifiers(x_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    models: Dict[str, object] = {}

    extra_trees = ExtraTreesClassifier(n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    extra_trees.fit(x_train, y_train)
    models["Extra Trees"] = extra_trees

    random_forest = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1)
    random_forest.fit(x_train, y_train)
    models["Random Forest"] = random_forest

    svm = SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    svm.fit(x_train, y_train)
    models["SVM"] = svm

    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(x_train, y_train)
    models["KNN"] = knn

    return models


def score_estimator(estimator, x_data: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_data)
        return normalize_scores(probabilities)

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(x_data)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])
        return softmax(decision)

    predictions = estimator.predict(x_data)
    one_hot = np.zeros((len(predictions), len(LABELS)), dtype=np.float32)
    for idx, prediction in enumerate(predictions):
        one_hot[idx, int(prediction)] = 1.0
    return one_hot


def normalize_scores(values: np.ndarray) -> np.ndarray:
    totals = np.sum(values, axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return values / totals


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def train_cnn_model(model: SimpleBrainCNN, records: Sequence[ImageRecord], device: torch.device) -> None:
    torch.manual_seed(RANDOM_STATE)
    loader = torch.utils.data.DataLoader(
        BrainTumorDataset(records, image_size=CNN_IMAGE_SIZE),
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(CNN_EPOCHS):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def generate_gradcam(model: SimpleBrainCNN, image: np.ndarray, target_class: int | None = None) -> np.ndarray:
    device = next(model.parameters()).device
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        del grad_input
        gradients.append(grad_output[0].detach())

    handle_fwd = model.gradcam_layer.register_forward_hook(forward_hook)
    handle_bwd = model.gradcam_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()
        image_tensor = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        logits = model(image_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.astype(np.float32)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def make_heatmap_overlay(source_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    base = source_image
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 0.55, colored, 0.45, 0)


def save_rgb_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path, format="PNG")
