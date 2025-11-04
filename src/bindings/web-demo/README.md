# NIMCP Web Demo v2.7.0

Interactive web-based demonstration of NIMCP cognitive learning capabilities with real-time visualization.

![NIMCP Web Demo](screenshot.png)

## Overview

### What Is This?

A full-stack web application showcasing NIMCP's neural learning capabilities through an interactive dashboard:

- **Real-time training visualization** - Watch loss decrease as the brain learns
- **Interactive predictions** - Test the trained brain with custom inputs
- **Metrics dashboard** - Monitor accuracy, performance, and statistics
- **Chart.js visualizations** - Beautiful real-time charts showing brain activity

### Why Was It Built?

To provide a comprehensive, production-ready reference implementation demonstrating:
- NIMCP Python bindings in action
- Modern web architecture with Flask + React
- Real-time data visualization with Chart.js
- Best practices for ML model serving

### How Does It Work?

- **Backend**: Flask REST API using NIMCP Python bindings
- **Frontend**: React app with Chart.js for visualization
- **Demo Task**: Iris flower classification (3 classes, 4 features)
- **Communication**: RESTful API with JSON payloads

---

## Features

### Backend (Flask)
- ✅ NIMCP brain initialization and management
- ✅ Single-example and batch training endpoints
- ✅ Real-time prediction with confidence scoring
- ✅ Metrics tracking (loss, accuracy, timing)
- ✅ Training/prediction history with sliding window
- ✅ CORS enabled for React frontend

### Frontend (React)
- ✅ **Training Panel**: Manual or quick-train with dataset
- ✅ **Prediction Panel**: Interactive inference with confidence visualization
- ✅ **Metrics Dashboard**: Real-time charts and statistics
- ✅ **Loss Chart**: Training progress over time
- ✅ **Confidence Chart**: Prediction confidence tracking
- ✅ **Class Distribution**: Bar chart of predictions by class
- ✅ **Recent Predictions Table**: Last 10 predictions with timing

---

## Quick Start

### Prerequisites

- Python 3.7+ with NIMCP bindings installed
- Node.js 14+ and npm
- NIMCP library built (`libnimcp.so`)

### Installation

#### 1. Backend Setup

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Ensure NIMCP is in Python path
export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH

# Start Flask server
python app.py
```

Backend will run on `http://localhost:5000`

#### 2. Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Start React development server
npm start
```

Frontend will run on `http://localhost:3000`

### Usage

1. **Open** `http://localhost:3000` in your browser
2. **Click** "Initialize Brain" to create NIMCP brain instance
3. **Train** using quick-train button or manual examples
4. **Predict** by entering flower measurements
5. **Watch** real-time metrics and charts update

---

## Architecture

### System Diagram

```
┌─────────────────┐         ┌──────────────────┐
│   React App     │  HTTP   │   Flask API      │
│  (Port 3000)    │ ◄─────► │   (Port 5000)    │
│                 │  JSON   │                  │
│ ┌─────────────┐ │         │ ┌──────────────┐ │
│ │Training     │ │         │ │NIMCP Brain   │ │
│ │Panel        │ │         │ │Python Module │ │
│ └─────────────┘ │         │ └──────────────┘ │
│ ┌─────────────┐ │         │                  │
│ │Prediction   │ │         │ libnimcp.so      │
│ │Panel        │ │         │                  │
│ └─────────────┘ │         └──────────────────┘
│ ┌─────────────┐ │
│ │Metrics      │ │
│ │Dashboard    │ │
│ └─────────────┘ │
└─────────────────┘
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/init` | POST | Initialize NIMCP brain |
| `/api/train` | POST | Train on single example |
| `/api/train-batch` | POST | Train on multiple examples |
| `/api/predict` | POST | Make prediction |
| `/api/metrics` | GET | Get metrics and history |
| `/api/status` | GET | Get brain status |
| `/api/dataset` | GET | Get iris dataset |
| `/api/reset` | POST | Reset brain and metrics |

### Data Flow

1. User interaction in React → API call to Flask
2. Flask processes request → Calls NIMCP Python bindings
3. NIMCP performs computation → Returns result
4. Flask updates metrics → Sends response to React
5. React updates UI → Charts re-render

---

## Directory Structure

```
web-demo/
├── backend/
│   ├── app.py              # Flask REST API
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML entry point
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # Styles
│   │   ├── index.js        # React entry point
│   │   └── components/
│   │       ├── TrainingPanel.js        # Training interface
│   │       ├── PredictionPanel.js      # Prediction interface
│   │       └── MetricsDashboard.js     # Charts and metrics
│   ├── package.json        # Node dependencies
│   └── README.md           # Frontend documentation
└── README.md               # This file
```

---

## Demo Task: Iris Classification

### Dataset

- **Classes**: Setosa, Versicolor, Virginica (3 iris flower species)
- **Features**:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Training samples**: 5 per class (15 total)

### Brain Configuration

- **Size**: SMALL (balanced performance/accuracy)
- **Task**: CLASSIFICATION
- **Inputs**: 4 (flower measurements)
- **Outputs**: 3 (flower species)

### Feature Normalization

Features are normalized to 0-1 range using min-max scaling:
- Sepal length: 4.0-8.0 cm
- Sepal width: 2.0-5.0 cm
- Petal length: 1.0-7.0 cm
- Petal width: 0.0-3.0 cm

---

## Metrics Tracked

### Training Metrics
- **Total Trained**: Number of training examples processed
- **Training Time**: Cumulative and average time per example
- **Loss History**: Training loss over time (last 50 examples)
- **Last Loss**: Most recent training loss

### Prediction Metrics
- **Total Predictions**: Number of predictions made
- **Prediction Time**: Cumulative and average inference time
- **Accuracy**: Percentage of correct predictions (when true label provided)
- **Confidence History**: Prediction confidence over time (last 20)
- **Class Distribution**: Count of predictions per class

---

## Development

### Backend Development

```bash
cd backend

# Run with debug mode
python app.py

# Test endpoints
curl http://localhost:5000/api/status
```

### Frontend Development

```bash
cd frontend

# Start development server (hot reload)
npm start

# Build for production
npm run build

# Run tests
npm test
```

### NIMCP Coding Standards Applied

All code follows NIMCP standards:
- ✅ **WHAT/WHY/HOW documentation** on all functions
- ✅ **Guard clauses** for input validation
- ✅ **Error handling** with exceptions and error messages
- ✅ **Memory safety** (no leaks, proper cleanup)
- ✅ **Type safety** with type hints (Python) and PropTypes (React)

---

## Production Deployment

### Docker (Recommended)

```dockerfile
# Dockerfile example
FROM python:3.10
WORKDIR /app
COPY backend/ /app/backend/
RUN pip install -r backend/requirements.txt
EXPOSE 5000
CMD ["python", "backend/app.py"]
```

### Nginx + Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run Flask with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 backend.app:app
```

### Environment Variables

```bash
# Backend
export FLASK_ENV=production
export NIMCP_LIB_PATH=/opt/nimcp/lib

# Frontend
export REACT_APP_API_URL=https://api.example.com
```

---

## Troubleshooting

### Backend Issues

**Problem**: "ModuleNotFoundError: No module named 'nimcp'"
```bash
# Solution: Ensure NIMCP is installed
cd /path/to/nimcp/src/bindings/python
python setup.py install --user
```

**Problem**: "libnimcp.so not found"
```bash
# Solution: Add library to path
export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH
```

### Frontend Issues

**Problem**: "Failed to fetch" / CORS errors
```bash
# Solution: Ensure Flask CORS is enabled and backend is running
# Check backend console for startup message
```

**Problem**: Charts not rendering
```bash
# Solution: Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

---

## Extending the Demo

### Add New Features

1. **New Endpoint**: Add to `backend/app.py`
2. **New Component**: Create in `frontend/src/components/`
3. **New Chart**: Use Chart.js in `MetricsDashboard.js`

### Different Dataset

Replace iris data in `app.py`:
```python
CUSTOM_DATA = {
    'class_a': [[...], [...], ...],
    'class_b': [[...], [...], ...]
}
```

Update brain configuration:
```python
brain = nimcp.Brain(
    name="custom_classifier",
    size=nimcp.BRAIN_MEDIUM,
    task=nimcp.TASK_CLASSIFICATION,
    num_inputs=YOUR_FEATURE_COUNT,
    num_outputs=YOUR_CLASS_COUNT
)
```

---

## Performance

### Expected Performance

- **Training**: ~1-5ms per example
- **Prediction**: ~0.5-2ms per inference
- **API Latency**: ~5-10ms round-trip
- **Frontend**: 60 FPS chart updates

### Optimization Tips

- Use batch training for faster initialization
- Enable SIMD operations in NIMCP build
- Use production React build (`npm run build`)
- Cache static assets with Nginx

---

## Screenshots

### Main Dashboard
![Dashboard](docs/dashboard.png)

### Training Panel
![Training](docs/training.png)

### Metrics Visualization
![Metrics](docs/metrics.png)

---

## License

MIT License - Same as NIMCP

---

## Support

- **Issues**: GitHub issue tracker
- **Documentation**: See `docs/` directory
- **Examples**: This demo is the reference implementation
- **Community**: NIMCP discussions forum

---

## Version History

- **v2.7.0** (2025-11-05): Initial release
  - Flask backend with REST API
  - React frontend with Chart.js
  - Iris classification demo
  - Real-time metrics dashboard
  - Comprehensive documentation

---

**Built with NIMCP v2.7.0 - Neural Interface Message Communication Protocol**
