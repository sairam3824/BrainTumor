import { useState, useCallback } from "react";
import { predictImage } from "../services/api.js";

/**
 * Custom hook managing the full prediction lifecycle:
 * file selection → upload → loading → result/error.
 */
export default function usePrediction() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);

  const selectFile = useCallback((selectedFile) => {
    setFile(selectedFile);
    setResult(null);
    setError(null);
    setProgress(0);

    // Generate preview URL
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(selectedFile);
    } else {
      setPreview(null);
    }
  }, []);

  const predict = useCallback(async () => {
    if (!file) return;

    setLoading(true);
    setError(null);
    setProgress(0);

    try {
      const data = await predictImage(file, setProgress);
      setResult(data);

      // Save to history in localStorage
      const history = JSON.parse(localStorage.getItem("prediction_history") || "[]");
      history.unshift({
        id: Date.now(),
        filename: file.name,
        prediction: data.prediction,
        confidence: data.confidence,
        is_tumor: data.is_tumor,
        timestamp: new Date().toISOString(),
      });
      // Keep last 50
      localStorage.setItem("prediction_history", JSON.stringify(history.slice(0, 50)));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [file]);

  const reset = useCallback(() => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setProgress(0);
  }, []);

  return {
    file,
    preview,
    result,
    loading,
    error,
    progress,
    selectFile,
    predict,
    reset,
  };
}
