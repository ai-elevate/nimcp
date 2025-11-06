import React, { useState, useEffect } from 'react'

const PATTERN_NAMES = ['Vertical |', 'Horizontal ─', 'Diagonal \\', 'Diagonal /']
const PATTERN_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']

function OutputVisualization({ outputActivations }) {
  const [predictedPattern, setPredictedPattern] = useState('none')
  const [maxActivation, setMaxActivation] = useState(0)

  useEffect(() => {
    if (outputActivations && outputActivations.length === 4) {
      const max = Math.max(...outputActivations)
      setMaxActivation(max)
      if (max > 0.1) {
        const predictedIndex = outputActivations.indexOf(max)
        setPredictedPattern(PATTERN_NAMES[predictedIndex])
      } else {
        setPredictedPattern('none')
      }
    }
  }, [outputActivations])

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
