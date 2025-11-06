import React, { useState } from 'react'

const PRESET_PATTERNS = {
  vertical: [0, 1, 0, 0, 1, 0, 0, 1, 0],
  horizontal: [0, 0, 0, 1, 1, 1, 0, 0, 0],
  diagonal_down: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  diagonal_up: [0, 0, 1, 0, 1, 0, 1, 0, 0]
}

function PatternInput({ onPresentPattern, onTrainPattern }) {
  const [pattern, setPattern] = useState(Array(9).fill(0))
  const [selectedPreset, setSelectedPreset] = useState('vertical')
  const [trainingLabel, setTrainingLabel] = useState(0)
  const [trainingIterations, setTrainingIterations] = useState(10)

  const toggleCell = (index) => {
    const newPattern = [...pattern]
    newPattern[index] = newPattern[index] === 1 ? 0 : 1
    setPattern(newPattern)
  }

  const loadPreset = (presetName) => {
    setSelectedPreset(presetName)
    setPattern([...PRESET_PATTERNS[presetName]])
  }

  const clearPattern = () => {
    setPattern(Array(9).fill(0))
  }

  const handlePresent = () => {
    onPresentPattern(pattern)
  }

  const handleTrain = () => {
    onTrainPattern(selectedPreset, trainingLabel, trainingIterations)
  }

  return (
    <div className="pattern-input-panel">
      <h3>🎨 Pattern Input</h3>

      {/* 3x3 Grid */}
      <div className="pattern-grid">
        {pattern.map((cell, index) => (
          <div
            key={index}
            className={`pattern-cell ${cell === 1 ? 'active' : ''}`}
            onClick={() => toggleCell(index)}
          >
            {cell === 1 && '●'}
          </div>
        ))}
      </div>

      {/* Preset Patterns */}
      <div className="preset-controls">
        <label>Load Preset:</label>
        <div className="preset-buttons">
          <button
            className="btn btn-sm"
            onClick={() => loadPreset('vertical')}
          >
            |
          </button>
          <button
            className="btn btn-sm"
            onClick={() => loadPreset('horizontal')}
          >
            ─
          </button>
          <button
            className="btn btn-sm"
            onClick={() => loadPreset('diagonal_down')}
          >
            \
          </button>
          <button
            className="btn btn-sm"
            onClick={() => loadPreset('diagonal_up')}
          >
            /
          </button>
          <button
            className="btn btn-sm btn-warning"
            onClick={clearPattern}
          >
            Clear
          </button>
        </div>
      </div>

      {/* Actions */}
      <div className="pattern-actions">
        <button
          className="btn btn-primary"
          onClick={handlePresent}
        >
          👁️ Present Pattern
        </button>

        <div className="training-controls">
          <label>Training:</label>
          <select
            value={selectedPreset}
            onChange={(e) => setSelectedPreset(e.target.value)}
          >
            <option value="vertical">Vertical |</option>
            <option value="horizontal">Horizontal ─</option>
            <option value="diagonal_down">Diagonal \</option>
            <option value="diagonal_up">Diagonal /</option>
          </select>

          <label>Label (Output Neuron):</label>
          <input
            type="number"
            min="0"
            max="3"
            value={trainingLabel}
            onChange={(e) => setTrainingLabel(parseInt(e.target.value))}
          />

          <label>Iterations:</label>
          <input
            type="number"
            min="1"
            max="100"
            value={trainingIterations}
            onChange={(e) => setTrainingIterations(parseInt(e.target.value))}
          />

          <button
            className="btn btn-success"
            onClick={handleTrain}
          >
            🎓 Train Pattern
          </button>
        </div>
      </div>
    </div>
  )
}

export default PatternInput
