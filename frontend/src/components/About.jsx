import React from "react";
import "./CSS/About.css";
import Omid from "./CSS/Images/Omid.png";
import Navbar from "./Navbar";
import Footer from "./Footer";

const About = () => {
  return (
    <div className="about-container">
                  <Navbar />

      <h1 className="about-title">About AirPiano</h1>
      <p className="about-description">
        <strong>AirPiano</strong> is a revolutionary platform designed to make music accessible to everyone. Whether you're a beginner or an expert, this tool empowers individuals, including disabled and underprivileged communities, to learn and play instruments for free, anywhere and anytime.
      </p>
      <p className="about-mission">
        Our mission is to bridge the gap in music education and bring the joy of music to everyone, irrespective of their background or circumstances.
      </p>
      <h2 className="developers-title">The Team(?)</h2>
      <div className="developers">
        <div className="developer">
          <img src={Omid} alt="Omid Latifi" className="developer-image" />
          <h3 className="developer-name">Omid Latifi</h3>
          <p className="developer-description">
            Omid is a passionate software developer committed to creating innovative solutions for real-world challenges. With expertise in web development and a love for music, he developed AirPiano to make music accessible to all.
          </p>
        </div>
      </div>
      <Footer />

    </div>
  );
};

export default About;