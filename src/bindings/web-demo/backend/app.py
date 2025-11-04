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

from flask import Flask, request, jsonify
from flask_cors import CORS
import nimcp
import time
import threading
import queue
from datetime import datetime

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
    print("="*70)

    app.run(debug=True, host='0.0.0.0', port=5000)
