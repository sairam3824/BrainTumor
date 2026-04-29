import { useState } from "react";
import { NavLink } from "react-router-dom";
import { Brain, Menu, X } from "lucide-react";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  const linkClass = ({ isActive }) =>
    `navbar-link${isActive ? " active" : ""}`;

  return (
    <nav className="navbar" id="main-nav">
      <div className="navbar-inner">
        <NavLink to="/" className="navbar-brand" onClick={() => setMenuOpen(false)}>
          <span className="navbar-brand-icon">
            <Brain size={20} color="#fff" />
          </span>
          NeuroScan AI
        </NavLink>

        <button
          className="mobile-menu-btn"
          onClick={() => setMenuOpen((v) => !v)}
          aria-label="Toggle menu"
          id="mobile-menu-toggle"
        >
          {menuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        <ul className={`navbar-links${menuOpen ? " open" : ""}`}>
          <li>
            <NavLink to="/" className={linkClass} onClick={() => setMenuOpen(false)} id="nav-home">
              Home
            </NavLink>
          </li>
          <li>
            <NavLink to="/dashboard" className={linkClass} onClick={() => setMenuOpen(false)} id="nav-dashboard">
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink to="/about" className={linkClass} onClick={() => setMenuOpen(false)} id="nav-about">
              About
            </NavLink>
          </li>
        </ul>
      </div>
    </nav>
  );
}
