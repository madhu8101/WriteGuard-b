import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import LoadingIndicator from '../components/LoadingIndicator';
import ClassificationResult from '../components/ClassificationResult';
import RecommendationList from '../components/RecommendationList';
import logo from '../logo.png';
import './Dashboard.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function Dashboard() {
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [filterType, setFilterType] = useState('all');
  const [loading, setLoading] = useState(false);
  const [classification, setClassification] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [extractedText, setExtractedText] = useState('');
  const [userEmail, setUserEmail] = useState('');

  useEffect(() => {
    const isLoggedIn = localStorage.getItem('isLoggedIn');
    if (!isLoggedIn) {
      navigate('/login');
      return;
    }
    setUserEmail(localStorage.getItem('userEmail') || 'User');
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('isLoggedIn');
    localStorage.removeItem('userEmail');
    navigate('/login');
  };

  const handleTextSubmit = async (e) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setClassification(null);
    setRecommendations([]);

    try {
      // Get classification
      const predictResponse = await axios.post(`${API_URL}/predict`, {
        text: text
      });
      setClassification(predictResponse.data);

      // Get recommendations
      const recommendResponse = await axios.get(`${API_URL}/recommendations`, {
        params: {
          text: text,
          filter_type: filterType,
          top_k: 5
        }
      });
      setRecommendations(recommendResponse.data.recommendations);
    } catch (err) {
      console.error('Error:', err);
      alert('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    // Validate file type
    const validTypes = ['.pdf', '.txt'];
    const fileExtension = uploadedFile.name.substring(uploadedFile.name.lastIndexOf('.')).toLowerCase();
    if (!validTypes.includes(fileExtension)) {
      alert('Please upload a PDF or TXT file.');
      return;
    }

    setLoading(true);
    setClassification(null);
    setRecommendations([]);
    setExtractedText('');

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setExtractedText(response.data.extracted_text);
      setClassification(response.data.classification);

      // Get recommendations for extracted text
      if (response.data.extracted_text) {
        const recommendResponse = await axios.get(`${API_URL}/recommendations`, {
          params: {
            text: response.data.extracted_text.substring(0, 1000), // Use first 1000 chars
            filter_type: filterType,
            top_k: 5
          }
        });
        setRecommendations(recommendResponse.data.recommendations);
      }
    } catch (err) {
      console.error('Error:', err);
      alert('An error occurred while processing the file.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setFile(null);
    setClassification(null);
    setRecommendations([]);
    setExtractedText('');
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-left">
          <img src={logo} alt="WriteGuard AI" className="header-logo" />
          <div className="header-title">
            <h1>WriteGuard AI</h1>
            <span className="user-info">Welcome, {userEmail}</span>
          </div>
        </div>
        <button className="logout-button" onClick={handleLogout}>
          Sign Out
        </button>
      </header>

      <main className="dashboard-main">
        <div className="input-section">
          <h2>Text Classification</h2>
          <p className="section-description">
            Enter text or upload a document to classify it as Human-written, AI-generated, or Humanized AI.
          </p>

          <form onSubmit={handleTextSubmit} className="text-input-form">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter or paste your text here for classification..."
              rows="6"
              disabled={loading}
            />
            <div className="form-actions">
              <button
                type="submit"
                className="submit-button"
                disabled={loading || !text.trim()}
              >
                {loading ? <LoadingIndicator size="small" /> : 'Classify Text'}
              </button>
              <button
                type="button"
                className="clear-button"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </button>
            </div>
          </form>

          <div className="upload-section">
            <h3>Or Upload a Document</h3>
            <div className="file-upload-container">
              <input
                type="file"
                id="file-upload"
                accept=".pdf,.txt"
                onChange={handleFileUpload}
                disabled={loading}
                className="file-input"
              />
              <label htmlFor="file-upload" className="file-upload-label">
                <span className="upload-icon">📄</span>
                <span>Choose PDF or TXT file</span>
              </label>
              {file && <span className="file-name">{file.name}</span>}
            </div>
          </div>

          <div className="filter-section">
            <h3>Recommendation Filter</h3>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              disabled={loading}
              className="filter-select"
            >
              <option value="all">All Articles</option>
              <option value="ai">AI-Generated</option>
              <option value="human">Human-Written</option>
              <option value="humanized">Humanized AI</option>
            </select>
          </div>
        </div>

        {loading && <LoadingIndicator size="large" />}

        {classification && (
          <ClassificationResult classification={classification} />
        )}

        {recommendations.length > 0 && (
          <RecommendationList recommendations={recommendations} />
        )}

        {extractedText && (
          <div className="extracted-text-section">
            <h3>Extracted Text from Document</h3>
            <div className="extracted-text">
              {extractedText.length > 2000
                ? `${extractedText.substring(0, 2000)}...`
                : extractedText}
            </div>
          </div>
        )}
      </main>

      <footer className="dashboard-footer">
        <p>AI vs Human vs Humanized Text Classification System &copy; 2026</p>
      </footer>
    </div>
  );
}

export default Dashboard;