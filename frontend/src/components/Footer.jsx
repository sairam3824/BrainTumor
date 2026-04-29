import { Heart } from "lucide-react";
import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="footer" id="footer">
      <p>
        Built with <Heart size={14} style={{ color: "#ef4444", verticalAlign: "middle" }} /> using Deep Learning
      </p>
      <div className="footer-links">
        <Link to="/" className="footer-link">Home</Link>
        <Link to="/dashboard" className="footer-link">Dashboard</Link>
        <Link to="/about" className="footer-link">About</Link>
      </div>
    </footer>
  );
}
