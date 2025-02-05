import React from "react";
import "./CSS/Footer.css";

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-container">
        <p className="footer-text">
          &copy; {new Date().getFullYear()} AirPiano | Designed by Omid Latifi
        </p>
        <div className="footer-links">
          <a href="/about" className="footer-link">About</a>
          <a href="#contact" className="footer-link">Contact</a>
          <a href="#privacy" className="footer-link">Privacy Policy</a>
        </div>
      </div>
    </footer>
  );
};

export default Footer;