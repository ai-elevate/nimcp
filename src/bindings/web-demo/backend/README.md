# NIMCP Web Demo - Backend

Flask REST API serving NIMCP brain for web interface.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure NIMCP is available
export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH

# Run server
python app.py
```

Server runs on `http://localhost:5000`

## API Documentation

### POST /api/init
Initialize NIMCP brain for iris classification.

**Request**: Empty
**Response**:
```json
{
  "success": true,
  "message": "Brain initialized successfully",
  "metrics": {...}
}
```

### POST /api/train
Train brain on single example.

**Request**:
```json
{
  "features": [5.1, 3.5, 1.4, 0.2],
  "label": "setosa",
  "confidence": 1.0
}
```

**Response**:
```json
{
  "success": true,
  "loss": 0.1234,
  "elapsed": 0.002,
  "metrics": {...}
}
```

### POST /api/train-batch
Train on multiple examples at once.

**Request**:
```json
{
  "examples": [
    {"features": [5.1, 3.5, 1.4, 0.2], "label": "setosa", "confidence": 1.0},
    {"features": [7.0, 3.2, 4.7, 1.4], "label": "versicolor", "confidence": 1.0}
  ]
}
```

### POST /api/predict
Make prediction on features.

**Request**:
```json
{
  "features": [5.0, 3.5, 1.5, 0.3],
  "true_label": "setosa"  // optional
}
```

**Response**:
```json
{
  "success": true,
  "prediction": "setosa",
  "confidence": 0.95,
  "correct": true,
  "elapsed": 0.001,
  "metrics": {...}
}
```

### GET /api/metrics
Get current metrics and history.

**Response**:
```json
{
  "success": true,
  "metrics": {
    "total_trained": 15,
    "total_predictions": 5,
    "accuracy_estimate": 80.0,
    "last_loss": 0.05,
    ...
  },
  "training_history": [...],
  "prediction_history": [...]
}
```

### GET /api/status
Check if brain is initialized.

**Response**:
```json
{
  "success": true,
  "initialized": true,
  "status": "training",
  "metrics": {...}
}
```

### GET /api/dataset
Get iris dataset for training.

**Response**:
```json
{
  "success": true,
  "dataset": {
    "setosa": [[5.1, 3.5, 1.4, 0.2], ...],
    "versicolor": [...],
    "virginica": [...]
  },
  "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
  "classes": ["setosa", "versicolor", "virginica"]
}
```

### POST /api/reset
Reset brain and clear all metrics.

**Response**:
```json
{
  "success": true,
  "message": "Brain reset successfully"
}
```

## Dependencies

- Flask 3.0.0 - Web framework
- flask-cors 4.0.0 - CORS support for React
- nimcp 2.7.0 - NIMCP Python bindings

## Configuration

Edit `app.py` to customize:

```python
# Change port
app.run(debug=True, host='0.0.0.0', port=8080)

# Change brain size
brain = nimcp.Brain(
    name="classifier",
    size=nimcp.BRAIN_MEDIUM,  # TINY, SMALL, MEDIUM, LARGE
    ...
)

# Change dataset
CUSTOM_DATA = {
    'class_a': [[...], ...],
    'class_b': [[...], ...]
}
```

## Error Handling

All endpoints return standard error format:

```json
{
  "success": false,
  "error": "Error message here"
}
```

HTTP status codes:
- 200: Success
- 400: Bad request (invalid parameters)
- 500: Internal server error

## Development

```bash
# Run with debug mode (auto-reload)
python app.py

# Test endpoints
curl -X POST http://localhost:5000/api/init
curl -X GET http://localhost:5000/api/status

# View logs
# Flask prints to console by default
```

## Production

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or use uWSGI
uwsgi --http :5000 --wsgi-file app.py --callable app
```
