import { Scan, RotateCcw } from "lucide-react";
import ImageUploader from "../components/ImageUploader.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ProbabilityChart from "../components/ProbabilityChart.jsx";
import usePrediction from "../hooks/usePrediction.js";

export default function HomePage() {
  const {
    file,
    preview,
    result,
    loading,
    error,
    selectFile,
    predict,
    reset,
  } = usePrediction();

  return (
    <div className="page-container" id="home-page">
      {/* Header */}
      <header className="page-header">
        <h1 className="page-title">Brain Tumor Classification</h1>
        <p className="page-subtitle">
          Upload an MRI scan and let our AI-powered model classify it instantly
          using deep learning and support vector machines.
        </p>
      </header>

      {/* Upload Section */}
      <div className="glass-card" style={{ padding: 32, maxWidth: 720, margin: "0 auto" }}>
        <ImageUploader
          onFileSelect={selectFile}
          preview={preview}
          file={file}
          disabled={loading}
        />

        {/* Action Buttons */}
        {file && !loading && (
          <div
            style={{
              display: "flex",
              gap: 12,
              justifyContent: "center",
              marginTop: 24,
            }}
          >
            <button
              className="btn-primary"
              onClick={predict}
              disabled={!file}
              id="predict-btn"
            >
              <Scan size={18} /> Analyze Scan
            </button>
            <button className="btn-secondary" onClick={reset} id="reset-btn">
              <RotateCcw size={18} /> Reset
            </button>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading-overlay" id="loading-indicator">
            <div className="spinner" />
            <p className="loading-text">Analyzing your MRI scan...</p>
            <p className="loading-subtext">
              Running through Autoencoder → PCA → SVM pipeline
            </p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="error-box" id="error-message">
            <span>⚠️</span>
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="result-section" id="results-section">
          <div className="result-grid">
            <ResultCard result={result} />
            <ProbabilityChart probabilities={result.probabilities} />
          </div>

          {/* Analyze another */}
          <div style={{ textAlign: "center", marginTop: 32 }}>
            <button className="btn-secondary" onClick={reset} id="analyze-another-btn">
              <RotateCcw size={18} /> Analyze Another Scan
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
