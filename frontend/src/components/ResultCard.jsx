import { ShieldCheck, ShieldAlert } from "lucide-react";
import { CLASS_CONFIG } from "../utils/constants.js";

export default function ResultCard({ result }) {
  if (!result) return null;

  const { prediction, confidence, is_tumor } = result;
  const config = CLASS_CONFIG[prediction] || {};

  return (
    <div className="result-main glass-card" id="result-card">
      {/* Badge */}
      <div className={`result-badge ${is_tumor ? "tumor" : "no-tumor"}`}>
        {is_tumor ? (
          <>
            <ShieldAlert size={14} /> Tumor Detected
          </>
        ) : (
          <>
            <ShieldCheck size={14} /> All Clear
          </>
        )}
      </div>

      {/* Class Name */}
      <h2 className="result-class" style={{ color: config.color }}>
        {config.emoji} {config.label}
      </h2>

      {/* Description */}
      <p style={{ color: "#94a3b8", fontSize: "0.9rem", marginBottom: 20 }}>
        {config.description}
      </p>

      {/* Confidence */}
      <p className="result-confidence">
        Confidence:{" "}
        <span className="confidence-value" style={{ color: config.color }}>
          {confidence.toFixed(1)}%
        </span>
      </p>
      <div className="confidence-bar-wrapper">
        <div
          className="confidence-bar"
          style={{ width: `${confidence}%` }}
        />
      </div>

      {/* Grad-CAM Heatmap */}
      {result.gradcam_image && (
        <div style={{ marginTop: '20px', textAlign: 'center', backgroundColor: 'rgba(255,255,255,0.05)', padding: '15px', borderRadius: '12px' }}>
          <h3 style={{ fontSize: '0.9rem', marginBottom: '10px', color: '#cbd5e1', fontWeight: 500 }}>
            Model Heatmap (Grad-CAM)
          </h3>
          <img 
            src={result.gradcam_image} 
            alt="Grad-CAM Heatmap" 
            style={{ 
              width: '100%', 
              maxWidth: '200px', 
              borderRadius: '8px', 
              boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
              border: '1px solid rgba(255,255,255,0.1)'
            }} 
          />
        </div>
      )}
    </div>
  );
}
