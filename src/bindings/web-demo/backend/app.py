"""
NIMCP Web Demo - Flask Backend v2.7.0
======================================

WHAT: REST API backend demonstrating NIMCP brain with real-time metrics
WHY:  Show NIMCP capabilities through interactive web interface
HOW:  Flask server exposing brain training/prediction endpoints with metrics

Features:
- Real-time training with progress tracking
- Interactive predictions
- Brain metrics and statistics
- Iris flower classification demo
"""

from flask import Flask, request, jsonify, send_file, redirect
from flask_cors import CORS
import nimcp
import time
import threading
import queue
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global brain instance and metrics
brain = None
brain_lock = threading.Lock()
training_history = []
prediction_history = []
brain_metrics = {
    'total_trained': 0,
    'total_predictions': 0,
    'training_time': 0.0,
    'prediction_time': 0.0,
    'last_loss': 0.0,
    'accuracy_estimate': 0.0,
    'created_at': None,
    'status': 'not_initialized'
}

# Iris dataset for demo
IRIS_DATA = {
    'setosa': [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
    ],
    'versicolor': [
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
    ],
    'virginica': [
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
    ]
}

#=============================================================================
# Helper Functions
#=============================================================================

def normalize_features(features):
    """
    WHAT: Normalize features to 0-1 range
    WHY:  Improve training stability and convergence
    HOW:  Min-max normalization based on typical iris ranges
    """
    # Typical ranges: sepal_length: 4-8, sepal_width: 2-5, petal_length: 1-7, petal_width: 0-3
    ranges = [(4.0, 8.0), (2.0, 5.0), (1.0, 7.0), (0.0, 3.0)]
    normalized = []
    for i, val in enumerate(features):
        min_val, max_val = ranges[i]
        normalized.append((val - min_val) / (max_val - min_val))
    return normalized

def calculate_accuracy():
    """
    WHAT: Estimate brain accuracy on recent predictions
    WHY:  Provide accuracy metric for dashboard
    HOW:  Count correct predictions in last 20 predictions
    """
    if len(prediction_history) < 5:
        return 0.0

    recent = prediction_history[-20:] if len(prediction_history) > 20 else prediction_history
    correct = sum(1 for p in recent if p.get('correct', False))
    return (correct / len(recent)) * 100.0

#=============================================================================
# API Endpoints
#=============================================================================

@app.route('/api/init', methods=['POST'])
def init_brain():
    """
    WHAT: Initialize NIMCP brain
    WHY:  Create brain instance for training/prediction
    HOW:  Call nimcp.Brain with iris classification config
    """
    global brain, brain_metrics

    try:
        with brain_lock:
            # Create brain for iris classification (4 inputs, 3 outputs)
            brain = nimcp.Brain(
                name="iris_classifier",
                size=nimcp.BRAIN_SMALL,
                task=nimcp.TASK_CLASSIFICATION,
                num_inputs=4,
                num_outputs=3
            )

            brain_metrics['created_at'] = datetime.now().isoformat()
            brain_metrics['status'] = 'initialized'
            brain_metrics['total_trained'] = 0
            brain_metrics['total_predictions'] = 0

        return jsonify({
            'success': True,
            'message': 'Brain initialized successfully',
            'metrics': brain_metrics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_brain():
    """
    WHAT: Train brain on single example
    WHY:  Update brain with new training data
    HOW:  Normalize features, call brain.learn(), track metrics
    """
    global brain, brain_metrics, training_history

    if brain is None:
        return jsonify({'success': False, 'error': 'Brain not initialized'}), 400

    try:
        data = request.json
        features = data.get('features')  # [sepal_length, sepal_width, petal_length, petal_width]
        label = data.get('label')        # 'setosa', 'versicolor', or 'virginica'
        confidence = data.get('confidence', 1.0)

        # Guard: Validate inputs
        if not features or not label:
            return jsonify({'success': False, 'error': 'Missing features or label'}), 400

        if len(features) != 4:
            return jsonify({'success': False, 'error': 'Expected 4 features'}), 400

        # Normalize features
        normalized = normalize_features(features)

        # Train
        start_time = time.time()
        with brain_lock:
            loss = brain.learn(normalized, label, confidence)
        elapsed = time.time() - start_time

        # Update metrics
        brain_metrics['total_trained'] += 1
        brain_metrics['training_time'] += elapsed
        brain_metrics['last_loss'] = float(loss)
        brain_metrics['status'] = 'training'

        # Record training history
        training_history.append({
            'timestamp': datetime.now().isoformat(),
            'loss': float(loss),
            'label': label,
            'elapsed': elapsed
        })

        # Keep only last 100 entries
        if len(training_history) > 100:
            training_history = training_history[-100:]

        return jsonify({
            'success': True,
            'loss': float(loss),
            'elapsed': elapsed,
            'metrics': brain_metrics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train-batch', methods=['POST'])
def train_batch():
    """
    WHAT: Train on multiple examples at once
    WHY:  Quickly initialize brain with dataset
    HOW:  Iterate through examples, train on each
    """
    global brain

    if brain is None:
        return jsonify({'success': False, 'error': 'Brain not initialized'}), 400

    try:
        data = request.json
        examples = data.get('examples', [])

        if not examples:
            return jsonify({'success': False, 'error': 'No examples provided'}), 400

        results = []
        for example in examples:
            features = example.get('features')
            label = example.get('label')
            confidence = example.get('confidence', 1.0)

            normalized = normalize_features(features)

            with brain_lock:
                loss = brain.learn(normalized, label, confidence)

            brain_metrics['total_trained'] += 1
            brain_metrics['last_loss'] = float(loss)

            results.append({
                'label': label,
                'loss': float(loss)
            })

            training_history.append({
                'timestamp': datetime.now().isoformat(),
                'loss': float(loss),
                'label': label,
                'elapsed': 0.0
            })

        # Keep only last 100 entries
        if len(training_history) > 100:
            training_history = training_history[-100:]

        return jsonify({
            'success': True,
            'trained_count': len(results),
            'results': results,
            'metrics': brain_metrics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    WHAT: Make prediction on features
    WHY:  Test trained brain, show inference
    HOW:  Normalize features, call brain.predict(), track metrics
    """
    global brain, brain_metrics, prediction_history

    if brain is None:
        return jsonify({'success': False, 'error': 'Brain not initialized'}), 400

    try:
        data = request.json
        features = data.get('features')
        true_label = data.get('true_label')  # Optional, for accuracy tracking

        # Guard: Validate inputs
        if not features:
            return jsonify({'success': False, 'error': 'Missing features'}), 400

        if len(features) != 4:
            return jsonify({'success': False, 'error': 'Expected 4 features'}), 400

        # Normalize features
        normalized = normalize_features(features)

        # Predict
        start_time = time.time()
        with brain_lock:
            predicted_label, confidence = brain.predict(normalized)
        elapsed = time.time() - start_time

        # Update metrics
        brain_metrics['total_predictions'] += 1
        brain_metrics['prediction_time'] += elapsed
        brain_metrics['status'] = 'predicting'

        # Check if prediction was correct
        correct = (predicted_label == true_label) if true_label else None

        # Record prediction history
        prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'predicted': predicted_label,
            'confidence': float(confidence),
            'true_label': true_label,
            'correct': correct,
            'elapsed': elapsed
        })

        # Keep only last 100 entries
        if len(prediction_history) > 100:
            prediction_history = prediction_history[-100:]

        # Update accuracy estimate
        brain_metrics['accuracy_estimate'] = calculate_accuracy()

        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': float(confidence),
            'correct': correct,
            'elapsed': elapsed,
            'metrics': brain_metrics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """
    WHAT: Get current brain metrics and history
    WHY:  Provide data for dashboard visualization
    HOW:  Return metrics, training/prediction history
    """
    return jsonify({
        'success': True,
        'metrics': brain_metrics,
        'training_history': training_history[-50:],  # Last 50 training examples
        'prediction_history': prediction_history[-20:]  # Last 20 predictions
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """
    WHAT: Get brain status
    WHY:  Check if brain is initialized and ready
    HOW:  Return current status and basic info
    """
    return jsonify({
        'success': True,
        'initialized': brain is not None,
        'status': brain_metrics['status'],
        'metrics': brain_metrics
    })

@app.route('/api/dataset', methods=['GET'])
def get_dataset():
    """
    WHAT: Get iris dataset for training
    WHY:  Provide demo data to frontend
    HOW:  Return IRIS_DATA dictionary
    """
    return jsonify({
        'success': True,
        'dataset': IRIS_DATA,
        'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'classes': list(IRIS_DATA.keys())
    })

@app.route('/api/reset', methods=['POST'])
def reset_brain():
    """
    WHAT: Reset brain and clear all metrics
    WHY:  Start fresh demo
    HOW:  Destroy brain, clear history, reset metrics
    """
    global brain, brain_metrics, training_history, prediction_history

    with brain_lock:
        brain = None
        training_history = []
        prediction_history = []
        brain_metrics = {
            'total_trained': 0,
            'total_predictions': 0,
            'training_time': 0.0,
            'prediction_time': 0.0,
            'last_loss': 0.0,
            'accuracy_estimate': 0.0,
            'created_at': None,
            'status': 'not_initialized'
        }

    return jsonify({
        'success': True,
        'message': 'Brain reset successfully'
    })

@app.route('/docs')
def documentation():
    """
    WHAT: Serve documentation page
    WHY:  Provide access to README and documentation
    HOW:  Render simple HTML page with links to docs
    """
    docs_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NIMCP Web Demo - Documentation</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
                max-width: 900px;
                margin: 50px auto;
                padding: 20px;
                line-height: 1.6;
                background: #0a0a0a;
                color: #e0e0e0;
            }
            h1 { color: #61dafb; border-bottom: 2px solid #61dafb; padding-bottom: 10px; }
            h2 { color: #61dafb; margin-top: 30px; }
            a { color: #61dafb; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .section {
                background: #1a1a1a;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                border-left: 4px solid #61dafb;
            }
            code {
                background: #2a2a2a;
                padding: 2px 6px;
                border-radius: 3px;
                color: #ffa500;
            }
            pre {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
            }
            .back-link {
                display: inline-block;
                margin: 20px 0;
                padding: 10px 20px;
                background: #61dafb;
                color: #0a0a0a;
                border-radius: 5px;
                font-weight: bold;
            }
            .back-link:hover {
                background: #4fa8c5;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <h1>🧠 NIMCP Web Demo Documentation</h1>

        <a href="/" class="back-link">← Back to Demo</a>

        <div class="section">
            <h2>What is NIMCP?</h2>
            <p><strong>NIMCP (Neural Inspired Model Control Protocol)</strong> is a biologically-inspired spiking neural network library that provides temporal pattern recognition and adaptive learning capabilities.</p>
        </div>

        <div class="section">
            <h2>📖 Quick Start</h2>
            <p>This demo showcases NIMCP's capabilities through an interactive Iris flower classification task.</p>
            <ol>
                <li>Click <strong>"Initialize Brain"</strong> to create a new neural network</li>
                <li>Use <strong>"Train on Dataset"</strong> to train the brain on iris flower samples</li>
                <li>Enter custom features or use presets to <strong>make predictions</strong></li>
                <li>Monitor real-time <strong>metrics and visualizations</strong> as the brain learns</li>
            </ol>
        </div>

        <div class="section">
            <h2>🔧 API Endpoints</h2>
            <pre><code>POST /api/init           - Initialize brain
POST /api/train          - Train on single example
POST /api/train-batch    - Train on multiple examples
POST /api/predict        - Make prediction
GET  /api/metrics        - Get metrics and history
GET  /api/status         - Get brain status
GET  /api/dataset        - Get iris dataset
POST /api/reset          - Reset brain</code></pre>
        </div>

        <div class="section">
            <h2>📊 Features</h2>
            <ul>
                <li><strong>Real-time Visualization</strong> - Watch loss decrease as the brain learns</li>
                <li><strong>Interactive Predictions</strong> - Test the trained brain with custom inputs</li>
                <li><strong>Metrics Dashboard</strong> - Monitor accuracy, performance, and statistics</li>
                <li><strong>Biologically Inspired</strong> - Spiking neural networks with STDP learning</li>
            </ul>
        </div>

        <div class="section">
            <h2>🔗 Additional Resources</h2>
            <ul>
                <li><a href="/api/docs/readme" target="_blank">Full README</a></li>
                <li><a href="/api/docs/quick-start" target="_blank">Quick Start Guide</a></li>
                <li><a href="https://github.com/bbrelin/nimcp" target="_blank">GitHub Repository</a></li>
            </ul>
        </div>

        <div class="section">
            <h2>💡 Example Usage</h2>
            <pre><code>// Initialize brain
fetch('/api/init', { method: 'POST' })

// Train on example
fetch('/api/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    features: [5.1, 3.5, 1.4, 0.2],
    label: 'setosa',
    confidence: 1.0
  })
})

// Make prediction
fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    features: [6.3, 3.3, 6.0, 2.5]
  })
})</code></pre>
        </div>

        <a href="/" class="back-link">← Back to Demo</a>

        <p style="text-align: center; margin-top: 50px; color: #666;">
            NIMCP v2.7.0 - Neural Inspired Model Control Protocol
        </p>
    </body>
    </html>
    """
    return docs_html

@app.route('/api/docs/readme')
def serve_readme():
    """
    WHAT: Serve README.md file
    WHY:  Provide detailed documentation
    HOW:  Send README.md from parent directory
    """
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'README.md')
    if os.path.exists(readme_path):
        return send_file(readme_path, mimetype='text/markdown', download_name='NIMCP-README.md')
    else:
        return jsonify({'success': False, 'error': 'README not found'}), 404

@app.route('/api/docs/quick-start')
def serve_quickstart():
    """
    WHAT: Serve QUICK_START.md file
    WHY:  Provide quick start guide
    HOW:  Send QUICK_START.md from parent directory
    """
    quickstart_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'QUICK_START.md')
    if os.path.exists(quickstart_path):
        return send_file(quickstart_path, mimetype='text/markdown', download_name='NIMCP-QUICK_START.md')
    else:
        return jsonify({'success': False, 'error': 'Quick Start guide not found'}), 404

#=============================================================================
# Main
#=============================================================================

if __name__ == '__main__':
    print("="*70)
    print("NIMCP Web Demo - Backend Server")
    print("="*70)
    print("Starting Flask server on http://localhost:5000")
    print("API Endpoints:")
    print("  POST /api/init           - Initialize brain")
    print("  POST /api/train          - Train on single example")
    print("  POST /api/train-batch    - Train on multiple examples")
    print("  POST /api/predict        - Make prediction")
    print("  GET  /api/metrics        - Get metrics and history")
    print("  GET  /api/status         - Get brain status")
    print("  GET  /api/dataset        - Get iris dataset")
    print("  POST /api/reset          - Reset brain")
    print("")
    print("Documentation:")
    print("  GET  /docs               - Interactive documentation page")
    print("  GET  /api/docs/readme    - Download README.md")
    print("  GET  /api/docs/quick-start - Download QUICK_START.md")
    print("="*70)

    app.run(debug=True, host='0.0.0.0', port=5000)
