/**
 * Centralized API service layer.
 * All backend communication goes through this module.
 */
import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:5000",
  timeout: 60000, // 60s for large model inference
  headers: {
    Accept: "application/json",
  },
});

// ── Response interceptor for uniform error handling ──
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.message ||
      "An unexpected error occurred";
    return Promise.reject(new Error(message));
  }
);

/**
 * Upload an MRI image for prediction.
 * @param {File} imageFile - The image file to classify
 * @param {function} onProgress - Optional upload progress callback (0-100)
 * @returns {Promise<Object>} prediction result
 */
export async function predictImage(imageFile, onProgress) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const response = await api.post("/api/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (event) => {
      if (onProgress && event.total) {
        onProgress(Math.round((event.loaded * 100) / event.total));
      }
    },
  });

  return response.data;
}

/**
 * Check backend health.
 * @returns {Promise<Object>}
 */
export async function checkHealth() {
  const response = await api.get("/api/health");
  return response.data;
}

/**
 * Get supported tumor classes and descriptions.
 * @returns {Promise<Object>}
 */
export async function getClasses() {
  const response = await api.get("/api/classes");
  return response.data;
}

export default api;
