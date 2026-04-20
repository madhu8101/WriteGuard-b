import React from 'react';
import './ClassificationResult.css';

function ClassificationResult({ classification }) {
  const { label, confidence, probabilities } = classification;

  const getLabelColor = (label) => {
    switch (label) {
      case 'human':
        return '#27ae60';
      case 'ai':
        return '#e74c3c';
      case 'humanized':
        return '#f39c12';
      default:
        return '#3498db';
    }
  };

  const getLabelIcon = (label) => {
    switch (label) {
      case 'human':
        return '👤';
      case 'ai':
        return '🤖';
      case 'humanized':
        return '👥';
      default:
        return '❓';
    }
  };

  const getLabelDescription = (label) => {
    switch (label) {
      case 'human':
        return 'This text appears to be written by a human.';
      case 'ai':
        return 'This text appears to be AI-generated.';
      case 'humanized':
        return 'This text appears to be AI-generated content that has been edited by a human.';
      default:
        return 'Classification result.';
    }
  };

  return (
    <div className="classification-result">
      <h2>Classification Result</h2>
      
      <div className="result-card">
        <div 
          className="result-label" 
          style={{ borderColor: getLabelColor(label) }}
        >
          <span className="result-icon">{getLabelIcon(label)}</span>
          <div className="result-info">
            <span className="result-category" style={{ color: getLabelColor(label) }}>
              {label.toUpperCase()}
            </span>
            <p className="result-description">
              {getLabelDescription(label)}
            </p>
          </div>
        </div>

        <div className="confidence-section">
          <h3>Confidence Score</h3>
          <div className="confidence-bar-container">
            <div 
              className="confidence-bar"
              style={{ 
                width: `${confidence * 100}%`,
                backgroundColor: getLabelColor(label)
              }}
            ></div>
          </div>
          <span className="confidence-value">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>

        {probabilities && (
          <div className="probabilities-section">
            <h3>Class Probabilities</h3>
            <div className="probability-bars">
              {Object.entries(probabilities).map(([cls, prob]) => (
                <div key={cls} className="probability-item">
                  <span className="probability-label">{cls}</span>
                  <div className="probability-bar-container">
                    <div 
                      className="probability-bar"
                      style={{ 
                        width: `${prob * 100}%`,
                        backgroundColor: cls === label ? getLabelColor(label) : '#bdc3c7'
                      }}
                    ></div>
                  </div>
                  <span className="probability-value">{(prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ClassificationResult;