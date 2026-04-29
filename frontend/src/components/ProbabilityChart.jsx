import { CLASS_CONFIG } from "../utils/constants.js";

export default function ProbabilityChart({ probabilities }) {
  if (!probabilities) return null;

  // Sort by probability descending
  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a);

  return (
    <div className="result-chart glass-card" id="probability-chart">
      <h3 className="chart-title">Class Probabilities</h3>
      <div className="prob-bars">
        {sorted.map(([className, prob]) => {
          const config = CLASS_CONFIG[className] || {};
          return (
            <div key={className} className="prob-item">
              <div className="prob-label-row">
                <span className="prob-label">
                  {config.emoji || ""} {config.label || className}
                </span>
                <span className="prob-value">{prob.toFixed(1)}%</span>
              </div>
              <div className="prob-bar-track">
                <div
                  className={`prob-bar-fill ${className}`}
                  style={{ width: `${prob}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
