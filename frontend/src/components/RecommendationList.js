import React from 'react';
import './RecommendationList.css';

function RecommendationList({ recommendations }) {
  const getTypeBadgeClass = (type) => {
    switch (type) {
      case 'ai':
        return 'badge-ai';
      case 'human':
        return 'badge-human';
      case 'humanized':
        return 'badge-humanized';
      default:
        return 'badge-default';
    }
  };

  const getTypeLabel = (type) => {
    switch (type) {
      case 'ai':
        return 'AI-Generated';
      case 'human':
        return 'Human-Written';
      case 'humanized':
        return 'Humanized AI';
      default:
        return type;
    }
  };

  return (
    <div className="recommendation-list">
      <h2>Recommended Research Articles</h2>
      <p className="recommendations-subtitle">
        Based on semantic similarity to your input text
      </p>

      <div className="recommendations-grid">
        {recommendations.map((rec, index) => (
          <div key={index} className="recommendation-card">
            <div className="card-header">
              <span className="card-number">#{index + 1}</span>
              <span className={`type-badge ${getTypeBadgeClass(rec.type)}`}>
                {getTypeLabel(rec.type)}
              </span>
            </div>

            <h3 className="card-title">{rec.title}</h3>

            <p className="card-content">
              {rec.content.length > 300
                ? `${rec.content.substring(0, 300)}...`
                : rec.content}
            </p>

            <div className="card-footer">
              <div className="similarity-score">
                <span className="similarity-label">Similarity:</span>
                <span className="similarity-value">
                  {(rec.similarity * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RecommendationList;