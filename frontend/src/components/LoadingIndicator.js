import React from 'react';
import './LoadingIndicator.css';

function LoadingIndicator({ size = 'medium', message = 'Loading...' }) {
  return (
    <div className={`loading-container loading-${size}`}>
      <div className="spinner"></div>
      {message && <p className="loading-message">{message}</p>}
    </div>
  );
}

export default LoadingIndicator;