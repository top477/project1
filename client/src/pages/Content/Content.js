import React, { useState } from "react";
import "./style.css";
import Foundation from "../../ContentPage/Foundation/Foundation";
import Beginner from "../../ContentPage/Beginner/Beginner";
import Intermediate from "../../ContentPage/Intermediate/Intermediate";
import Advance from "../../ContentPage/Advance/Advance";
import Progress from "../../components/Progress/Progress";
import { useNavigate } from "react-router-dom";
import DownloadIcon from "@mui/icons-material/Download";

const Content = () => {
  // State to track the currently active topic
  const [activeTopic, setActiveTopic] = useState(null);
  
  // State to store the selected content paragraph
  const [contentParagraph, setContentParagraph] = useState("");
  
  // Hook to handle navigation between routes
  const navigate = useNavigate();

  // Function to toggle the active topic
  const toggleTopic = (topic) => {
    // Clear the content paragraph when switching topics
    setContentParagraph("");
    
    // Set the active topic, or deactivate if the same topic is clicked
    setActiveTopic(activeTopic === topic ? null : topic);
  };

  // Function to display content for the selected topic
  const displayContent = (content) => {
    // Update the content paragraph with the selected content
    setContentParagraph(content);
  };

  return (
    <div className="content">
      {/* Left section with topics and progress */}
      <div id="c1">
        <h1>Topic</h1>
        <div className="sub-topic">
          <ul>
            {/* Foundation topic and its sub-topics */}
            <li>
              <span onClick={() => toggleTopic("foundation")}>Foundation</span>
              {activeTopic === "foundation" && (
                <ul>
                  <li onClick={() => displayContent("How Do I Get Started?")}>
                    How Do I Get Started?
                  </li>
                  <li onClick={() => displayContent("Step-by-Step Process")}>
                    Step-by-Step Process
                  </li>
                  <li onClick={() => displayContent("Probability")}>
                    Probability
                  </li>
                  <li onClick={() => displayContent("Statistical Methods")}>
                    Statistical Methods
                  </li>
                  <li onClick={() => displayContent("Linear Algebra")}>
                    Linear Algebra
                  </li>
                  <li onClick={() => displayContent("Optimization")}>
                    Optimization
                  </li>
                  <li onClick={() => displayContent("Calculus")}>Calculus</li>
                  <li onClick={() => displayContent("Quiz")}>Quiz</li>
                </ul>
              )}
            </li>
            
            {/* Beginner topic and its sub-topics */}
            <li>
              <span onClick={() => toggleTopic("beginner")}>Beginner</span>
              {activeTopic === "beginner" && (
                <ul>
                  <li onClick={() => displayContent("Python Skills")}>
                    Python Skills
                  </li>
                  <li onClick={() => displayContent("Understand ML Algorithms")}>
                    Understand ML Algorithms
                  </li>
                  <li onClick={() => displayContent("ML + Weka (no code)")}>
                    ML + Weka (no code)
                  </li>
                  <li onClick={() => displayContent("ML + Python (scikit-learn)")}>
                    ML + Python (scikit-learn)
                  </li>
                  <li onClick={() => displayContent("ML + R (caret)")}>
                    ML + R (caret)
                  </li>
                  <li onClick={() => displayContent("Time Series Forecasting")}>
                    Time Series Forecasting
                  </li>
                  <li onClick={() => displayContent("Data Preparation")}>
                    Data Preparation
                  </li>
                  <li onClick={() => displayContent("Quiz")}>Quiz</li>
                </ul>
              )}
            </li>
            
            {/* Intermediate topic and its sub-topics */}
            <li>
              <span onClick={() => toggleTopic("intermediate")}>Intermediate</span>
              {activeTopic === "intermediate" && (
                <ul>
                  <li onClick={() => displayContent("Code ML Algorithms")}>
                    Code ML Algorithms
                  </li>
                  <li onClick={() => displayContent("XGBoost Algorithm")}>
                    XGBoost Algorithm
                  </li>
                  <li onClick={() => displayContent("Imbalanced Classification")}>
                    Imbalanced Classification
                  </li>
                  <li onClick={() => displayContent("Deep Learning (Keras)")}>
                    Deep Learning (Keras)
                  </li>
                  <li onClick={() => displayContent("Deep Learning (PyTorch)")}>
                    Deep Learning (PyTorch)
                  </li>
                  <li onClick={() => displayContent("ML in OpenCV")}>
                    ML in OpenCV
                  </li>
                  <li onClick={() => displayContent("Better Deep Learning")}>
                    Better Deep Learning
                  </li>
                  <li onClick={() => displayContent("Ensemble Learning")}>
                    Ensemble Learning
                  </li>
                  <li onClick={() => displayContent("Quiz")}>Quiz</li>
                </ul>
              )}
            </li>
            
            {/* Advance topic and its sub-topics */}
            <li>
              <span onClick={() => toggleTopic("advance")}>Advance</span>
              {activeTopic === "advance" && (
                <ul>
                  <li onClick={() => displayContent("Long Short-Term Memory")}>
                    Long Short-Term Memory
                  </li>
                  <li onClick={() => displayContent("Natural Language (Text)")}>
                    Natural Language (Text)
                  </li>
                  <li onClick={() => displayContent("Computer Vision")}>
                    Computer Vision
                  </li>
                  <li onClick={() => displayContent("CNN/LSTM + Time Series")}>
                    CNN/LSTM + Time Series
                  </li>
                  <li onClick={() => displayContent("GANs")}>GANs</li>
                  <li onClick={() => displayContent("Attention and Transformers")}>
                    Attention and Transformers
                  </li>
                  <li onClick={() => displayContent("Quiz")}>Quiz</li>
                </ul>
              )}
            </li>
          </ul>
        </div>
        
        {/* Progress section with a download icon leading to the certificate page */}
        <div className="progress">
          <DownloadIcon
            onClick={() => navigate("/certificate")}
            style={{ marginLeft: "70px", padding: "10px" }}
          />
          <Progress />
        </div>
      </div>

      {/* Right section to display content based on the active topic */}
      <div id="c2">
        {activeTopic === "foundation" && <Foundation topic={contentParagraph} />}
        {activeTopic === "beginner" && <Beginner topic={contentParagraph} />}
        {activeTopic === "intermediate" && (
          <Intermediate topic={contentParagraph} />
        )}
        {activeTopic === "advance" && <Advance topic={contentParagraph} />}
      </div>
    </div>
  );
};

export default Content;
