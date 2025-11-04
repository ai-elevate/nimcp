/**
 * TrainingPanel Component
 * =======================
 *
 * WHAT: Interface for training the NIMCP brain
 * WHY:  Allow users to train on individual or batch examples
 * HOW:  Forms for manual input or quick-train with dataset
 */

import React, { useState } from 'react';

function TrainingPanel({ onTrain, onTrainBatch, dataset, loading }) {
  const [sepalLength, setSepalLength] = useState('5.1');
  const [sepalWidth, setSepalWidth] = useState('3.5');
  const [petalLength, setPetalLength] = useState('1.4');
  const [petalWidth, setPetalWidth] = useState('0.2');
  const [selectedClass, setSelectedClass] = useState('setosa');
  const [confidence, setConfidence] = useState('1.0');
  const [trainingStatus, setTrainingStatus] = useState('');

  const handleTrain = async (e) => {
    e.preventDefault();
    setTrainingStatus('Training...');

    try {
      const features = [
        parseFloat(sepalLength),
        parseFloat(sepalWidth),
        parseFloat(petalLength),
        parseFloat(petalWidth)
      ];

      const result = await onTrain(features, selectedClass, parseFloat(confidence));
      setTrainingStatus(`✅ Trained! Loss: ${result.loss.toFixed(4)}`);

      setTimeout(() => setTrainingStatus(''), 3000);
    } catch (err) {
      setTrainingStatus(`❌ Error: ${err.message}`);
    }
  };

  const handleQuickTrain = async () => {
    if (!dataset) return;

    setTrainingStatus('Quick training...');

    // Create training batch from dataset
    const examples = [];
    Object.entries(dataset).forEach(([label, samples]) => {
      samples.forEach(features => {
        examples.push({ features, label, confidence: 1.0 });
      });
    });

    try {
      await onTrainBatch(examples);
      setTrainingStatus(`✅ Trained on ${examples.length} examples!`);
      setTimeout(() => setTrainingStatus(''), 3000);
    } catch (err) {
      setTrainingStatus(`❌ Error: ${err.message}`);
    }
  };

  const loadExample = (className) => {
    if (!dataset || !dataset[className]) return;

    const example = dataset[className][0];
    setSepalLength(example[0].toString());
    setSepalWidth(example[1].toString());
    setPetalLength(example[2].toString());
    setPetalWidth(example[3].toString());
    setSelectedClass(className);
  };

  return (
    <div className="panel training-panel">
      <h2>🎓 Training Panel</h2>

      <div className="quick-train">
        <button
          className="btn btn-success"
          onClick={handleQuickTrain}
          disabled={loading || !dataset}
        >
          ⚡ Quick Train (All Dataset)
        </button>
        <p className="help-text">
          Train on all 15 iris examples at once
        </p>
      </div>

      <div className="divider">
        <span>OR</span>
      </div>

      <form onSubmit={handleTrain}>
        <h3>Train Single Example</h3>

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
          <label>Class</label>
          <select
            value={selectedClass}
            onChange={(e) => setSelectedClass(e.target.value)}
          >
            <option value="setosa">🌸 Setosa</option>
            <option value="versicolor">🌺 Versicolor</option>
            <option value="virginica">🌷 Virginica</option>
          </select>
        </div>

        <div className="form-group">
          <label>Confidence (0.0 - 1.0)</label>
          <input
            type="number"
            step="0.1"
            min="0"
            max="1"
            value={confidence}
            onChange={(e) => setConfidence(e.target.value)}
            required
          />
        </div>

        <div className="button-group">
          <button type="submit" className="btn btn-primary" disabled={loading}>
            Train Example
          </button>
        </div>

        {trainingStatus && (
          <div className="status-message">{trainingStatus}</div>
        )}
      </form>

      <div className="example-buttons">
        <p className="help-text">Load example:</p>
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
    </div>
  );
}

export default TrainingPanel;
