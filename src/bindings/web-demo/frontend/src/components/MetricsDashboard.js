/**
 * MetricsDashboard Component
 * ===========================
 *
 * WHAT: Real-time visualization of brain metrics with Chart.js
 * WHY:  Show training progress, prediction accuracy, and performance
 * HOW:  Line charts for loss/accuracy, bar charts for statistics
 */

import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function MetricsDashboard({ metrics, trainingHistory, predictionHistory }) {
  if (!metrics) {
    return (
      <div className="dashboard">
        <p>No metrics available yet. Initialize and train the brain first.</p>
      </div>
    );
  }

  // Prepare training loss chart data
  const lossChartData = {
    labels: trainingHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Training Loss',
        data: trainingHistory.map(entry => entry.loss),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const lossChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#e0e0e0' }
      },
      title: {
        display: true,
        text: 'Training Loss Over Time',
        color: '#ffffff',
        font: { size: 16 }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Training Iteration',
          color: '#e0e0e0'
        },
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        title: {
          display: true,
          text: 'Loss',
          color: '#e0e0e0'
        },
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        beginAtZero: true
      }
    }
  };

  // Prepare prediction confidence chart data
  const confidenceChartData = {
    labels: predictionHistory.map((_, i) => i + 1),
    datasets: [
      {
        label: 'Prediction Confidence',
        data: predictionHistory.map(entry => entry.confidence * 100),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        fill: true,
        tension: 0.4,
      }
    ]
  };

  const confidenceChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#e0e0e0' }
      },
      title: {
        display: true,
        text: 'Prediction Confidence',
        color: '#ffffff',
        font: { size: 16 }
      }
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Prediction Number',
          color: '#e0e0e0'
        },
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        title: {
          display: true,
          text: 'Confidence (%)',
          color: '#e0e0e0'
        },
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        beginAtZero: true,
        max: 100
      }
    }
  };

  // Count predictions by class
  const classCounts = predictionHistory.reduce((acc, entry) => {
    acc[entry.predicted] = (acc[entry.predicted] || 0) + 1;
    return acc;
  }, {});

  const classChartData = {
    labels: Object.keys(classCounts),
    datasets: [
      {
        label: 'Predictions by Class',
        data: Object.values(classCounts),
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 206, 86, 0.6)'
        ],
        borderColor: [
          'rgb(255, 99, 132)',
          'rgb(54, 162, 235)',
          'rgb(255, 206, 86)'
        ],
        borderWidth: 1
      }
    ]
  };

  const classChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      title: {
        display: true,
        text: 'Predictions by Class',
        color: '#ffffff',
        font: { size: 16 }
      }
    },
    scales: {
      x: {
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      y: {
        ticks: { color: '#e0e0e0' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' },
        beginAtZero: true
      }
    }
  };

  return (
    <div className="dashboard">
      <h2>📊 Metrics Dashboard</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-icon">🎓</div>
          <div className="metric-content">
            <div className="metric-label">Total Trained</div>
            <div className="metric-value">{metrics.total_trained}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🔮</div>
          <div className="metric-content">
            <div className="metric-label">Total Predictions</div>
            <div className="metric-value">{metrics.total_predictions}</div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">📈</div>
          <div className="metric-content">
            <div className="metric-label">Accuracy</div>
            <div className="metric-value">
              {metrics.accuracy_estimate.toFixed(1)}%
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">⚡</div>
          <div className="metric-content">
            <div className="metric-label">Last Loss</div>
            <div className="metric-value">
              {metrics.last_loss.toFixed(4)}
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">⏱️</div>
          <div className="metric-content">
            <div className="metric-label">Avg Training Time</div>
            <div className="metric-value">
              {metrics.total_trained > 0
                ? ((metrics.training_time / metrics.total_trained) * 1000).toFixed(2)
                : '0.00'} ms
            </div>
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-icon">🚀</div>
          <div className="metric-content">
            <div className="metric-label">Avg Prediction Time</div>
            <div className="metric-value">
              {metrics.total_predictions > 0
                ? ((metrics.prediction_time / metrics.total_predictions) * 1000).toFixed(2)
                : '0.00'} ms
            </div>
          </div>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          {trainingHistory.length > 0 ? (
            <Line data={lossChartData} options={lossChartOptions} />
          ) : (
            <div className="no-data">No training data yet</div>
          )}
        </div>

        <div className="chart-container">
          {predictionHistory.length > 0 ? (
            <Line data={confidenceChartData} options={confidenceChartOptions} />
          ) : (
            <div className="no-data">No prediction data yet</div>
          )}
        </div>

        <div className="chart-container">
          {predictionHistory.length > 0 ? (
            <Bar data={classChartData} options={classChartOptions} />
          ) : (
            <div className="no-data">No class distribution data yet</div>
          )}
        </div>
      </div>

      {predictionHistory.length > 0 && (
        <div className="recent-predictions">
          <h3>Recent Predictions</h3>
          <div className="predictions-table">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Predicted</th>
                  <th>Confidence</th>
                  <th>Correct</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {predictionHistory.slice(-10).reverse().map((pred, i) => (
                  <tr key={i}>
                    <td>{predictionHistory.length - i}</td>
                    <td>{pred.predicted}</td>
                    <td>{(pred.confidence * 100).toFixed(1)}%</td>
                    <td>
                      {pred.correct === null ? '-' : pred.correct ? '✅' : '❌'}
                    </td>
                    <td>{(pred.elapsed * 1000).toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default MetricsDashboard;
