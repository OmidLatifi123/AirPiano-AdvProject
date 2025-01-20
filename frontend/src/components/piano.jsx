import React, { useEffect, useState } from "react";
import axios from "axios";

// Piano Component
const Piano = () => {
  const [handData, setHandData] = useState([]);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("Initializing...");

  // Start the backend piano service on component load
  useEffect(() => {
    const startPianoBackend = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/start-piano");
        console.log(response.data.message);
        setBackendStatus(response.data.message);
      } catch (err) {
        console.error("Failed to start piano backend:", err);
        setBackendStatus("Failed to start piano backend. Ensure the backend is running.");
      }
    };

    startPianoBackend();
  }, []);

  // Fetch hand data from the backend periodically
  useEffect(() => {
    const fetchHandData = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:5000/hand-data");
        setHandData(response.data.hands || []);
      } catch (err) {
        console.error("Error fetching hand data:", err);
        setError("Could not fetch hand data. Ensure the backend is running.");
      }
    };

    const interval = setInterval(fetchHandData, 100); // Poll every 100ms
    return () => clearInterval(interval);
  }, []);

  // Render piano keys
  const renderKeys = () => {
    const notes = ["C", "D", "E", "F", "G", "A", "B", "C_high"];
    return notes.map((note, index) => (
      <div
        key={index}
        style={{
          display: "inline-block",
          width: "50px",
          height: "200px",
          margin: "2px",
          backgroundColor: "#fff",
          border: "1px solid #000",
          textAlign: "center",
          lineHeight: "200px",
        }}
      >
        {note}
      </div>
    ));
  };

  // Render detected hand data
  const renderHandData = () => {
    if (handData.length === 0) return <p>No hands detected.</p>;
    return handData.map((hand, index) => (
      <div key={index} style={{ margin: "10px", padding: "10px", border: "1px solid #ccc" }}>
        <h4>Hand {index + 1}</h4>
        <ul>
          {hand.map((landmark, idx) => (
            <li key={idx}>
              {`Landmark ${idx}: X=${landmark.x.toFixed(2)}, Y=${landmark.y.toFixed(2)}, Z=${landmark.z.toFixed(2)}`}
            </li>
          ))}
        </ul>
      </div>
    ));
  };

  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: "20px" }}>
      <h1>Digital Instrument - Virtual Piano</h1>
      <p style={{ color: backendStatus.includes("Failed") ? "red" : "green" }}>{backendStatus}</p>
      <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start" }}>
        {/* Webcam feed */}
        <iframe
          src="http://127.0.0.1:5000/webcam"
          style={{
            width: "320px",
            height: "240px",
            border: "1px solid black",
            marginRight: "20px",
          }}
          title="Webcam Feed"
        ></iframe>
        {/* Piano keys */}
        <div>
          <h3>Piano Keys</h3>
          <div>{renderKeys()}</div>
        </div>
      </div>
      {/* Hand data */}
      <div>
        <h3>Hand Data</h3>
        {error ? <p style={{ color: "red" }}>{error}</p> : renderHandData()}
      </div>
    </div>
  );
};

export default Piano;
