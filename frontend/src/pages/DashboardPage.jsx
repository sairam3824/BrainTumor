import { useState, useEffect } from "react";
import HistoryTable from "../components/HistoryTable.jsx";
import { BarChart3, Brain, ShieldCheck, Activity } from "lucide-react";

export default function DashboardPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    setHistory(stored);
  }, []);

  const clearHistory = () => {
    localStorage.removeItem("prediction_history");
    setHistory([]);
  };

  // Compute stats
  const totalScans = history.length;
  const tumorsFound = history.filter((h) => h.is_tumor).length;
  const clearScans = totalScans - tumorsFound;
  const avgConfidence =
    totalScans > 0
      ? (history.reduce((sum, h) => sum + h.confidence, 0) / totalScans).toFixed(1)
      : "—";

  const stats = [
    { icon: <BarChart3 size={22} />, value: totalScans, label: "Total Scans", bg: "rgba(6,182,212,0.1)", color: "#06b6d4" },
    { icon: <Brain size={22} />, value: tumorsFound, label: "Tumors Found", bg: "rgba(239,68,68,0.1)", color: "#ef4444" },
    { icon: <ShieldCheck size={22} />, value: clearScans, label: "Clear Scans", bg: "rgba(16,185,129,0.1)", color: "#10b981" },
    { icon: <Activity size={22} />, value: `${avgConfidence}%`, label: "Avg Confidence", bg: "rgba(139,92,246,0.1)", color: "#8b5cf6" },
  ];

  return (
    <div className="page-container" id="dashboard-page">
      <header className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">
          Track your scan history and prediction statistics at a glance.
        </p>
      </header>

      {/* Stats Cards */}
      <div className="stats-grid">
        {stats.map((s) => (
          <div key={s.label} className="stat-card glass-card">
            <div
              style={{
                width: 44,
                height: 44,
                borderRadius: 10,
                background: s.bg,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                margin: "0 auto 12px",
                color: s.color,
              }}
            >
              {s.icon}
            </div>
            <div className="stat-value">{s.value}</div>
            <div className="stat-label">{s.label}</div>
          </div>
        ))}
      </div>

      {/* History Table */}
      <HistoryTable history={history} onClear={clearHistory} />
    </div>
  );
}
