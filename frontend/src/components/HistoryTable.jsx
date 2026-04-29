import { Trash2, ClipboardList } from "lucide-react";
import { formatTimestamp } from "../utils/constants.js";

export default function HistoryTable({ history, onClear }) {
  if (!history || history.length === 0) {
    return (
      <div className="empty-state" id="empty-history">
        <div className="empty-state-icon">
          <ClipboardList size={48} strokeWidth={1} color="#64748b" />
        </div>
        <p>No predictions yet.</p>
        <p style={{ fontSize: "0.8rem", marginTop: 4 }}>
          Upload an MRI scan on the Home page to get started.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h3 style={{ fontWeight: 600 }}>Prediction History</h3>
        <button
          className="btn-secondary"
          onClick={onClear}
          style={{ padding: "8px 16px", fontSize: "0.8rem" }}
          id="clear-history-btn"
        >
          <Trash2 size={14} /> Clear All
        </button>
      </div>

      <div className="history-table-wrapper glass-card">
        <table className="history-table" id="history-table">
          <thead>
            <tr>
              <th>#</th>
              <th>File</th>
              <th>Prediction</th>
              <th>Confidence</th>
              <th>Status</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {history.map((item, index) => (
              <tr key={item.id}>
                <td>{index + 1}</td>
                <td style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {item.filename}
                </td>
                <td>
                  <span className={`class-chip ${item.prediction}`}>
                    {item.prediction.replace("_", " ")}
                  </span>
                </td>
                <td style={{ fontWeight: 600 }}>{item.confidence.toFixed(1)}%</td>
                <td>{item.is_tumor ? "⚠️ Tumor" : "✅ Clear"}</td>
                <td>{formatTimestamp(item.timestamp)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
