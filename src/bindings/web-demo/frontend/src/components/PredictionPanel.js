/**
 * PredictionPanel Component
 * ==========================
 *
 * WHAT: Interface for making predictions with trained brain
 * WHY:  Test brain inference and visualize confidence
 * HOW:  Input form with prediction results display
 */

import React, { useState } from 'react';

function PredictionPanel({ onPredict, dataset, loading }) {
  const [sepalLength, setSepalLength] = useState('5.0');
  const [sepalWidth, setSepalWidth] = useState('3.5');
  const [petalLength, setPetalLength] = useState('1.5');
  const [petalWidth, setPetalWidth] = useState('0.3');
  const [trueLabel, setTrueLabel] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predicting, setPredicting] = useState(false);

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredicting(true);
    setPredictionResult(null);

    try {
      const features = [
        parseFloat(sepalLength),
        parseFloat(sepalWidth),
        parseFloat(petalLength),
        parseFloat(petalWidth)
      ];

      const result = await onPredict(features, trueLabel || null);
      setPredictionResult(result);
    } catch (err) {
      setPredictionResult({
        success: false,
        error: err.message
      });
    } finally {
      setPredicting(false);
    }
  };

  const loadExample = (className) => {
    if (!dataset || !dataset[className]) return;

    // Load last example (different from training examples)
    const examples = dataset[className];
    const example = examples[examples.length - 1];

    setSepalLength(example[0].toString());
    setSepalWidth(example[1].toString());
    setPetalLength(example[2].toString());
    setPetalWidth(example[3].toString());
    setTrueLabel(className);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#10b981'; // Green
    if (confidence >= 0.6) return '#f59e0b'; // Orange
    return '#ef4444'; // Red
  };

  return (
    <div className="panel prediction-panel">
      <h2>🔮 Prediction Panel</h2>

      <form onSubmit={handlePredict}>
        <div className="form-row">
          <div className="form-group">
            <label>Sepal Length (cm)</label>
            <input
              type="number"
              step="0.1"
              value={sepalLength}
              onChange={(e) => setSepalLength(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Sepal Width (cm)</label>
            <input
              type="number"
              step="0.1"
              value={sepalWidth}
              onChange={(e) => setSepalWidth(e.target.value)}
              required
            />
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label>Petal Length (cm)</label>
            <input
              type="number"
              step="0.1"
              value={petalLength}
              onChange={(e) => setPetalLength(e.target.value)}
              required
            />
          </div>

          <div className="form-group">
            <label>Petal Width (cm)</label>
            <input
              type="number"
              step="0.1"
              value={petalWidth}
              onChange={(e) => setPetalWidth(e.target.value)}
              required
            />
          </div>
        </div>

        <div className="form-group">
          <label>True Label (optional, for accuracy tracking)</label>
          <select
            value={trueLabel}
            onChange={(e) => setTrueLabel(e.target.value)}
          >
            <option value="">-- Select if known --</option>
            <option value="setosa">🌸 Setosa</option>
            <option value="versicolor">🌺 Versicolor</option>
            <option value="virginica">🌷 Virginica</option>
          </select>
        </div>

        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || predicting}
        >
          {predicting ? 'Predicting...' : 'Make Prediction'}
        </button>
      </form>

      <div className="example-buttons">
        <p className="help-text">Load test example:</p>
        <button
          className="btn btn-outline btn-small"
          onClick={() => loadExample('setosa')}
        >
          Setosa
        </button>
        <button
          className="btn btn-outline btn-small"
          onClick={() => loadExample('versicolor')}
        >
          Versicolor
        </button>
        <button
          className="btn btn-outline btn-small"
          onClick={() => loadExample('virginica')}
        >
          Virginica
        </button>
      </div>

      {predictionResult && (
        <div className="prediction-result">
          {predictionResult.success ? (
            <>
              <h3>Prediction Result</h3>
              <div className="result-card">
                <div className="result-item">
                  <span className="result-label">Predicted Class:</span>
                  <span className="result-value prediction-class">
                    {predictionResult.prediction === 'setosa' && '🌸'}
                    {predictionResult.prediction === 'versicolor' && '🌺'}
                    {predictionResult.prediction === 'virginica' && '🌷'}
                    {' '}
                    {predictionResult.prediction.charAt(0).toUpperCase() +
                     predictionResult.prediction.slice(1)}
                  </span>
                </div>

                <div className="result-item">
                  <span className="result-label">Confidence:</span>
                  <div className="confidence-bar-container">
                    <div
                      className="confidence-bar"
                      style={{
                        width: `${predictionResult.confidence * 100}%`,
                        backgroundColor: getConfidenceColor(predictionResult.confidence)
                      }}
                    />
                    <span className="confidence-text">
                      {(predictionResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {predictionResult.correct !== null && (
                  <div className="result-item">
                    <span className="result-label">Accuracy:</span>
                    <span className={`result-value ${predictionResult.correct ? 'correct' : 'incorrect'}`}>
                      {predictionResult.correct ? '✅ Correct' : '❌ Incorrect'}
                    </span>
                  </div>
                )}

                <div className="result-item">
                  <span className="result-label">Inference Time:</span>
                  <span className="result-value">
                    {(predictionResult.elapsed * 1000).toFixed(2)} ms
                  </span>
                </div>
              </div>
            </>
          ) : (
            <div className="error-message">
              ❌ {predictionResult.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default PredictionPanel;
