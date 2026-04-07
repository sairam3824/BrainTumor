const state = {
  ready: false,
  selectedFile: null,
  defaultModel: null,
  models: [],
};

const els = {
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  previewImage: document.getElementById("preview-image"),
  previewEmpty: document.getElementById("preview-empty"),
  modelSelect: document.getElementById("model-select"),
  predictForm: document.getElementById("predict-form"),
  predictButton: document.getElementById("predict-button"),
  statusTitle: document.getElementById("status-title"),
  statusMessage: document.getElementById("status-message"),
  statusPill: document.getElementById("status-pill"),
  predictionLabel: document.getElementById("prediction-label"),
  predictionSubtitle: document.getElementById("prediction-subtitle"),
  modelUsed: document.getElementById("model-used"),
  scores: document.getElementById("scores"),
  gradcamImage: document.getElementById("gradcam-image"),
  gradcamNote: document.getElementById("gradcam-note"),
  bestModelNote: document.getElementById("best-model-note"),
  accuracyNote: document.getElementById("accuracy-note"),
};

function setStatus(payload) {
  const status = payload.status || "unknown";
  els.statusTitle.textContent = status === "ready" ? "Ready for inference" : "Preparing runtime assets";
  els.statusMessage.textContent = payload.message || "Loading…";
  els.statusPill.textContent = status;
  state.ready = status === "ready";
  updateButtonState();
}

function updateButtonState() {
  els.predictButton.disabled = !(state.ready && state.selectedFile);
}

function renderModels(payload) {
  state.models = payload.models || [];
  state.defaultModel = payload.default_model || null;

  els.modelSelect.innerHTML = "";
  state.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.label;
    if (model.name === state.defaultModel) {
      option.selected = true;
    }
    els.modelSelect.appendChild(option);
  });

  els.bestModelNote.textContent = payload.default_model_label || payload.default_model || "Unavailable";
  els.accuracyNote.textContent = payload.default_model_accuracy || "Unavailable";
}

function renderPreview(file) {
  const reader = new FileReader();
  reader.onload = () => {
    els.previewImage.src = reader.result;
    els.previewImage.style.display = "block";
    els.previewEmpty.style.display = "none";
  };
  reader.readAsDataURL(file);
}

function setFile(file) {
  state.selectedFile = file;
  renderPreview(file);
  updateButtonState();
}

function clearResults() {
  els.predictionLabel.textContent = "Running inference…";
  els.predictionSubtitle.textContent = "Generating classifier output and Grad-CAM.";
  els.modelUsed.textContent = "Working…";
  els.scores.innerHTML = "";
  els.gradcamImage.removeAttribute("src");
  els.gradcamImage.style.display = "none";
  els.gradcamNote.textContent = "Generating Grad-CAM overlay.";
}

function renderScores(scores) {
  els.scores.innerHTML = "";
  scores.forEach((entry) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <div class="score-label">${entry.label}</div>
      <div class="score-bar"><div class="score-fill" style="width:${entry.percent}%;"></div></div>
      <div class="score-value">${entry.percent.toFixed(1)}%</div>
    `;
    els.scores.appendChild(row);
  });
}

function renderResult(payload) {
  els.predictionLabel.textContent = payload.predicted_label;
  els.predictionSubtitle.textContent = payload.prediction_note;
  els.modelUsed.textContent = payload.model_label;
  renderScores(payload.scores || []);

  if (payload.gradcam_overlay) {
    els.gradcamImage.src = payload.gradcam_overlay;
    els.gradcamImage.style.display = "block";
    els.gradcamNote.textContent = payload.gradcam_note;
  } else {
    els.gradcamImage.style.display = "none";
    els.gradcamNote.textContent = "Grad-CAM was unavailable for this request.";
  }
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.message || "Request failed.");
  }
  return payload;
}

async function pollHealth() {
  try {
    const payload = await fetchJSON("/api/health");
    const wasReady = state.ready;
    setStatus(payload);
    if (state.ready && !wasReady) {
      loadModels();
    }
    if (!state.ready) {
      window.setTimeout(pollHealth, 3000);
    }
  } catch (error) {
    setStatus({ status: "error", message: error.message });
  }
}

async function loadModels() {
  try {
    const payload = await fetchJSON("/api/models");
    renderModels(payload);
  } catch (error) {
    els.bestModelNote.textContent = error.message;
    els.accuracyNote.textContent = "Unavailable";
  }
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    els.fileInput.click();
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  els.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove("is-dragover");
  });
});

els.dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    setFile(file);
  }
});

els.fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) {
    setFile(file);
  }
});

els.predictForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!(state.ready && state.selectedFile)) {
    return;
  }

  clearResults();
  els.predictButton.disabled = true;

  const formData = new FormData();
  formData.append("file", state.selectedFile);
  formData.append("model", els.modelSelect.value);

  try {
    const payload = await fetchJSON("/api/predict", {
      method: "POST",
      body: formData,
    });
    renderResult(payload);
  } catch (error) {
    els.predictionLabel.textContent = "Prediction failed";
    els.predictionSubtitle.textContent = error.message;
    els.modelUsed.textContent = "Unavailable";
  } finally {
    updateButtonState();
  }
});

pollHealth();
loadModels();
