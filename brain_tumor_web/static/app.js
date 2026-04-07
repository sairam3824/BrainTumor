const body = document.body;

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.message || "Request failed.");
  }
  return payload;
}

function renderScores(container, scores) {
  if (!container) return;
  container.innerHTML = "";
  (scores || []).forEach((entry) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <div class="score-label">${entry.label}</div>
      <div class="score-bar"><div class="score-fill" style="width:${entry.percent}%;"></div></div>
      <div class="score-value">${entry.percent.toFixed(1)}%</div>
    `;
    container.appendChild(row);
  });
}

function setButtonDisabled(button, disabled) {
  if (!button) return;
  button.disabled = disabled;
}

function initAnalyzePage() {
  const state = {
    ready: false,
    selectedFile: null,
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
    statusProgress: document.getElementById("status-progress"),
    statusProgressLabel: document.getElementById("status-progress-label"),
    predictionLabel: document.getElementById("prediction-label"),
    predictionSubtitle: document.getElementById("prediction-subtitle"),
    modelUsed: document.getElementById("model-used"),
    scores: document.getElementById("scores"),
    gradcamImage: document.getElementById("gradcam-image"),
    gradcamEmpty: document.getElementById("gradcam-empty"),
    gradcamNote: document.getElementById("gradcam-note"),
    formFeedback: document.getElementById("form-feedback"),
  };

  function updateButtonState() {
    setButtonDisabled(els.predictButton, !(state.ready && state.selectedFile));
  }

  function setStatus(payload) {
    const status = payload.status || "unknown";
    const progress = Number(payload.progress || 0);
    els.statusTitle.textContent = status === "ready" ? "Analysis runtime ready" : "Preparing runtime assets";
    els.statusMessage.textContent = payload.message || "Loading runtime assets.";
    els.statusPill.textContent = status;
    els.statusProgress.style.width = `${progress}%`;
    els.statusProgressLabel.textContent = `${progress}%`;
    state.ready = status === "ready";
    els.formFeedback.textContent = state.ready
      ? "Runtime ready. You can upload an image and run prediction."
      : "The runtime is still preparing assets, so prediction remains disabled.";
    updateButtonState();
  }

  function renderModels(payload) {
    els.modelSelect.innerHTML = "";
    (payload.models || []).forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.label;
      if (model.name === payload.default_model) {
        option.selected = true;
      }
      els.modelSelect.appendChild(option);
    });
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

  function clearResult() {
    els.predictionLabel.textContent = "Running analysis…";
    els.predictionSubtitle.textContent = "Generating probabilities, Grad-CAM, and a persistent report.";
    els.modelUsed.textContent = "Working…";
    els.scores.innerHTML = "";
    els.gradcamImage.removeAttribute("src");
    els.gradcamImage.style.display = "none";
    els.gradcamEmpty.hidden = false;
    els.gradcamNote.textContent = "Creating the Grad-CAM explainer overlay.";
  }

  function renderResult(payload) {
    els.predictionLabel.textContent = payload.predicted_label;
    els.predictionSubtitle.textContent = payload.prediction_note;
    els.modelUsed.textContent = payload.model_label;
    renderScores(els.scores, payload.scores || []);
    if (payload.gradcam_image_url) {
      els.gradcamImage.src = payload.gradcam_image_url;
      els.gradcamImage.style.display = "block";
      els.gradcamEmpty.hidden = true;
    } else {
      els.gradcamImage.style.display = "none";
      els.gradcamEmpty.hidden = false;
    }
    els.gradcamNote.textContent = payload.gradcam_note || "Grad-CAM unavailable.";
  }

  async function pollHealth() {
    try {
      const payload = await fetchJSON("/api/health");
      const wasReady = state.ready;
      setStatus(payload);
      if (state.ready && !wasReady) {
        await loadModels();
      }
      if (!state.ready) {
        window.setTimeout(pollHealth, 2500);
      }
    } catch (error) {
      setStatus({ status: "error", message: error.message, progress: 100 });
    }
  }

  async function loadModels() {
    try {
      const payload = await fetchJSON("/api/models");
      renderModels(payload);
    } catch (_error) {}
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
    if (!(state.ready && state.selectedFile)) return;

    clearResult();
    setButtonDisabled(els.predictButton, true);
    const formData = new FormData();
    formData.append("file", state.selectedFile);
    formData.append("model", els.modelSelect.value);

    try {
      const payload = await fetchJSON("/api/predict", { method: "POST", body: formData });
      renderResult(payload);
    } catch (error) {
      els.predictionLabel.textContent = "Prediction failed";
      els.predictionSubtitle.textContent = error.message;
      els.modelUsed.textContent = "Unavailable";
      els.gradcamNote.textContent = "No report was created.";
    } finally {
      updateButtonState();
    }
  });

  pollHealth();
  loadModels();
}

async function initHistoryPage() {
  const list = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  if (!list) return;

  try {
    const payload = await fetchJSON("/api/history");
    const items = payload.items || [];
    if (!items.length) {
      empty.hidden = false;
      list.innerHTML = "";
      return;
    }

    empty.hidden = true;
    list.innerHTML = "";
    items.forEach((item) => {
      const card = document.createElement("article");
      card.className = "glass-card history-card";
      card.innerHTML = `
        <div class="history-card__media">
          ${
            item.input_image_url || item.gradcam_image_url
              ? `
                <div class="history-media-grid">
                  ${
                    item.input_image_url
                      ? `
                        <figure class="history-media-tile">
                          <a href="${item.input_image_url}" target="_blank" rel="noopener noreferrer">
                            <img src="${item.input_image_url}" alt="Uploaded MRI preview">
                          </a>
                          <figcaption>Input</figcaption>
                        </figure>
                      `
                      : ""
                  }
                  ${
                    item.gradcam_image_url
                      ? `
                        <figure class="history-media-tile">
                          <a href="${item.gradcam_image_url}" target="_blank" rel="noopener noreferrer">
                            <img src="${item.gradcam_image_url}" alt="Grad-CAM preview">
                          </a>
                          <figcaption>Grad-CAM</figcaption>
                        </figure>
                      `
                      : ""
                  }
                </div>
              `
              : ""
          }
        </div>
        <div class="history-card__body">
          <p class="card-label">${item.created_at}</p>
          <h2>${item.predicted_label}</h2>
          <p>Model: ${item.selected_model} | Confidence: ${item.top_confidence_percent}%</p>
          <p class="muted-note">Analysis ID: ${item.analysis_id}</p>
        </div>
      `;
      list.appendChild(card);
    });
  } catch (error) {
    empty.hidden = false;
    empty.querySelector("p").textContent = error.message;
    list.innerHTML = "";
  }
}

if (body.dataset.page === "analyze") {
  initAnalyzePage();
}

if (body.dataset.page === "history") {
  initHistoryPage();
}
