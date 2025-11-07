import React, { useState, useEffect } from 'react'

const PATTERN_NAMES = ['Vertical |', 'Horizontal ─', 'Diagonal \\', 'Diagonal /']
const PATTERN_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

function OutputVisualization({ outputActivations }) {
  const [predictedPattern, setPredictedPattern] = useState('none')
  const [maxActivation, setMaxActivation] = useState(0)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7)
  const [feedbackResult, setFeedbackResult] = useState(null)
  const [predictedIndex, setPredictedIndex] = useState(-1)

  useEffect(() => {
    if (outputActivations && outputActivations.length === 4) {
      const max = Math.max(...outputActivations)
      setMaxActivation(max)
      if (max > 0.1) {
        const idx = outputActivations.indexOf(max)
        setPredictedIndex(idx)
        setPredictedPattern(PATTERN_NAMES[idx])
      } else {
        setPredictedIndex(-1)
        setPredictedPattern('none')
      }
    }
  }, [outputActivations])

  const meetsThreshold = maxActivation >= confidenceThreshold

  const applyFeedback = async (isCorrect) => {
    try {
      const response = await fetch('/api/reinforcement/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confidence_threshold: confidenceThreshold,
          correct_output: predictedIndex, // Using predicted index for simplicity
          is_correct: isCorrect
        })
      })

      const result = await response.json()

      if (result.success) {
        setFeedbackResult({
          action: result.action,
          reward: result.reward_signal,
          confidence: result.confidence,
          meetsThreshold: result.meets_threshold
        })

        // Clear feedback after 3 seconds
        setTimeout(() => setFeedbackResult(null), 3000)
      }
    } catch (error) {
      console.error('Error applying feedback:', error)
    }
  }

  const getActivationPercentage = (activation) => {
    return Math.min(100, Math.max(0, activation * 100))
  }

  return (
    <div className="output-visualization-panel">
      <h3>🎯 Output Layer</h3>

      <div className="prediction-display">
        <div className="prediction-label">Predicted Pattern:</div>
        <div className={`prediction-value ${maxActivation > 0.1 ? 'active' : ''}`}>
          {predictedPattern}
        </div>
        <div className={`confidence-indicator ${meetsThreshold ? 'high' : 'low'}`}>
          Confidence: {(maxActivation * 100).toFixed(1)}%
          {!meetsThreshold && predictedIndex >= 0 && (
            <span className="confidence-warning"> ⚠️ Below threshold</span>
          )}
        </div>
      </div>

      {/* Confidence Threshold Control */}
      <div className="confidence-threshold-control">
        <label>
          Confidence Threshold: <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong>
        </label>
        <input
          type="range"
          min="0"
          max="100"
          value={confidenceThreshold * 100}
          onChange={(e) => setConfidenceThreshold(e.target.value / 100)}
          className="threshold-slider"
        />
        <div className="threshold-marks">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>

      <div className="output-neurons">
        {outputActivations && outputActivations.map((activation, index) => {
          const percentage = getActivationPercentage(activation)
          const isMax = activation === maxActivation && maxActivation > 0.1

          return (
            <div key={index} className="output-neuron-container">
              <div className="output-neuron-label">
                {PATTERN_NAMES[index]}
              </div>

              <div className="output-neuron-bar-container">
                <div
                  className={`output-neuron-bar ${isMax ? 'predicted' : ''}`}
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: PATTERN_COLORS[index],
                    opacity: isMax ? 1 : 0.6
                  }}
                />
                <div className="output-neuron-value">
                  {activation.toFixed(3)}
                </div>
              </div>

              <div className={`output-neuron-indicator ${isMax ? 'active' : ''}`}>
                {isMax && '✓'}
              </div>
            </div>
          )
        })}
      </div>

      {/* Reinforcement Learning Feedback */}
      {predictedIndex >= 0 && (
        <div className="feedback-controls">
          <h4>Provide Feedback:</h4>
          <div className="feedback-buttons">
            <button
              className="btn btn-success btn-sm"
              onClick={() => applyFeedback(true)}
            >
              ✓ Correct
            </button>
            <button
              className="btn btn-danger btn-sm"
              onClick={() => applyFeedback(false)}
            >
              ✗ Incorrect
            </button>
          </div>
        </div>
      )}

      {/* Feedback Result */}
      {feedbackResult && (
        <div className={`feedback-result ${feedbackResult.reward > 0 ? 'reward' : 'punish'}`}>
          <strong>
            {feedbackResult.action === 'reward_strong' && '🎉 Strong Reward Applied'}
            {feedbackResult.action === 'reward_weak' && '👍 Weak Reward Applied'}
            {feedbackResult.action === 'punish_strong' && '⚠️ Strong Correction Applied'}
            {feedbackResult.action === 'punish_weak' && '📝 Weak Correction Applied'}
          </strong>
          <div className="feedback-details">
            Reward Signal: {feedbackResult.reward > 0 ? '+' : ''}{feedbackResult.reward.toFixed(2)} |
            Confidence: {(feedbackResult.confidence * 100).toFixed(1)}% |
            {feedbackResult.meetsThreshold ? ' ✓ Met threshold' : ' ⚠️ Below threshold'}
          </div>
        </div>
      )}

      <div className="output-legend">
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: PATTERN_COLORS[0] }}></div>
          <span>Vertical</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: PATTERN_COLORS[1] }}></div>
          <span>Horizontal</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: PATTERN_COLORS[2] }}></div>
          <span>Diagonal \</span>
        </div>
        <div className="legend-item">
          <div className="legend-color" style={{ backgroundColor: PATTERN_COLORS[3] }}></div>
          <span>Diagonal /</span>
        </div>
      </div>
    </div>
  )
}

export default OutputVisualization
