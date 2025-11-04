/**
 * NIMCP Web Demo - React Frontend v2.7.0
 * =======================================
 *
 * WHAT: Interactive web interface for NIMCP brain demonstration
 * WHY:  Visualize brain training, predictions, and metrics in real-time
 * HOW:  React app with Chart.js, communicating with Flask backend
 */

import React, { useState, useEffect } from 'react';
import TrainingPanel from './components/TrainingPanel';
import PredictionPanel from './components/PredictionPanel';
import MetricsDashboard from './components/MetricsDashboard';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:5000/api';

function App() {
  const [brainInitialized, setBrainInitialized] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Poll metrics every 2 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      if (brainInitialized) {
        fetchMetrics();
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [brainInitialized]);

  // Fetch dataset on mount
  useEffect(() => {
    fetchDataset();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API_URL}/metrics`);
      if (response.data.success) {
        setMetrics(response.data.metrics);
        setTrainingHistory(response.data.training_history);
        setPredictionHistory(response.data.prediction_history);
      }
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  const fetchDataset = async () => {
    try {
      const response = await axios.get(`${API_URL}/dataset`);
      if (response.data.success) {
        setDataset(response.data.dataset);
      }
    } catch (err) {
      console.error('Failed to fetch dataset:', err);
    }
  };

  const initializeBrain = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/init`);
      if (response.data.success) {
        setBrainInitialized(true);
        setMetrics(response.data.metrics);
      }
    } catch (err) {
      setError(`Failed to initialize brain: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const resetBrain = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_URL}/reset`);
      setBrainInitialized(false);
      setMetrics(null);
      setTrainingHistory([]);
      setPredictionHistory([]);
    } catch (err) {
      setError(`Failed to reset brain: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const trainExample = async (features, label, confidence = 1.0) => {
    try {
      const response = await axios.post(`${API_URL}/train`, {
        features,
        label,
        confidence
      });
      if (response.data.success) {
        await fetchMetrics();
        return response.data;
      }
    } catch (err) {
      setError(`Training failed: ${err.message}`);
      throw err;
    }
  };

  const trainBatch = async (examples) => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/train-batch`, {
        examples
      });
      if (response.data.success) {
        await fetchMetrics();
      }
    } catch (err) {
      setError(`Batch training failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const makePrediction = async (features, trueLabel = null) => {
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        features,
        true_label: trueLabel
      });
      if (response.data.success) {
        await fetchMetrics();
        return response.data;
      }
    } catch (err) {
      setError(`Prediction failed: ${err.message}`);
      throw err;
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🧠 NIMCP Web Demo v2.7.0</h1>
        <p>Interactive Neural Brain Visualization</p>
      </header>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      <div className="container">
        {!brainInitialized ? (
          <div className="init-panel">
            <h2>Initialize NIMCP Brain</h2>
            <p>Create a brain instance for Iris flower classification</p>
            <ul>
              <li>4 input features (sepal length/width, petal length/width)</li>
              <li>3 output classes (Setosa, Versicolor, Virginica)</li>
              <li>Small brain size with adaptive learning</li>
            </ul>
            <button
              className="btn btn-primary btn-large"
              onClick={initializeBrain}
              disabled={loading}
            >
              {loading ? 'Initializing...' : '🚀 Initialize Brain'}
            </button>
          </div>
        ) : (
          <>
            <div className="status-bar">
              <div className="status-item">
                <span className="status-label">Status:</span>
                <span className="status-value status-active">
                  {metrics?.status || 'Active'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Trained:</span>
                <span className="status-value">{metrics?.total_trained || 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Predictions:</span>
                <span className="status-value">{metrics?.total_predictions || 0}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Accuracy:</span>
                <span className="status-value">
                  {metrics?.accuracy_estimate?.toFixed(1) || 0}%
                </span>
              </div>
              <button
                className="btn btn-danger btn-small"
                onClick={resetBrain}
                disabled={loading}
              >
                Reset
              </button>
            </div>

            <div className="panels-grid">
              <TrainingPanel
                onTrain={trainExample}
                onTrainBatch={trainBatch}
                dataset={dataset}
                loading={loading}
              />

              <PredictionPanel
                onPredict={makePrediction}
                dataset={dataset}
                loading={loading}
              />
            </div>

            <MetricsDashboard
              metrics={metrics}
              trainingHistory={trainingHistory}
              predictionHistory={predictionHistory}
            />
          </>
        )}
      </div>

      <footer className="App-footer">
        <p>
          NIMCP v2.7.0 - Neural Interface Message Communication Protocol
        </p>
        <p>
          <a href="https://github.com/youruser/nimcp" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
          {' · '}
          <a href="/docs" target="_blank" rel="noopener noreferrer">
            Documentation
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
