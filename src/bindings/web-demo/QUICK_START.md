# NIMCP Web Demo - Quick Start Guide

Get the demo running in 5 minutes!

## Prerequisites

- ✅ Python 3.7+ with NIMCP bindings installed
- ✅ Node.js 14+ and npm
- ✅ NIMCP library built (`libnimcp.so` in `/path/to/nimcp/bin/`)

## Step-by-Step Setup

### 1. Verify NIMCP Installation

```bash
# Check if NIMCP Python module is available
python3 -c "import nimcp; print(nimcp.__version__)"

# Should print: 2.7.0 (or similar)
```

If this fails:
```bash
cd /path/to/nimcp/src/bindings/python
python3 setup.py install --user
```

### 2. Start Backend (Terminal 1)

```bash
# Navigate to backend directory
cd /path/to/nimcp/src/bindings/web-demo/backend

# Install Python dependencies
pip install -r requirements.txt

# Set library path (if needed)
export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH

# Start Flask server
python3 app.py
```

You should see:
```
======================================================================
NIMCP Web Demo - Backend Server
======================================================================
Starting Flask server on http://localhost:5000
...
 * Running on http://0.0.0.0:5000
```

### 3. Start Frontend (Terminal 2)

```bash
# Navigate to frontend directory
cd /path/to/nimcp/src/bindings/web-demo/frontend

# Install Node dependencies (first time only)
npm install

# Start React development server
npm start
```

Browser should automatically open to `http://localhost:3000`

### 4. Use the Demo

1. **Click "Initialize Brain"** - Creates NIMCP brain instance
2. **Click "Quick Train"** - Trains on all 15 iris examples
3. **Click "Make Prediction"** - Tests brain with example data
4. **Watch the charts update** - Real-time visualization

## Troubleshooting

### Backend won't start

**Error**: `ModuleNotFoundError: No module named 'nimcp'`
```bash
# Solution: Install NIMCP Python bindings
cd /path/to/nimcp/src/bindings/python
python3 setup.py install --user
```

**Error**: `libnimcp.so: cannot open shared object file`
```bash
# Solution: Add library to path
export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH

# Or add to ~/.bashrc for permanent fix
echo 'export LD_LIBRARY_PATH=/path/to/nimcp/bin:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Frontend won't start

**Error**: `npm: command not found`
```bash
# Solution: Install Node.js
# Ubuntu/Debian:
sudo apt install nodejs npm

# macOS:
brew install node

# Or download from: https://nodejs.org/
```

**Error**: Dependencies installation fails
```bash
# Solution: Clear cache and retry
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Demo Issues

**Problem**: "Failed to fetch" errors in browser console
```bash
# Solution: Ensure backend is running
# Check terminal 1 for Flask server output
# Backend should be on http://localhost:5000
```

**Problem**: Charts not showing
```bash
# Solution: Refresh page or clear browser cache
# Try different browser (Chrome/Firefox recommended)
```

## Next Steps

Once running:

### Try Training
- Use "Quick Train" to train on all examples
- Watch the training loss chart decrease
- Try manual training with custom measurements

### Test Predictions
- Load example data with class buttons
- Enter custom flower measurements:
  - Sepal length: 4.0-8.0 cm
  - Sepal width: 2.0-5.0 cm
  - Petal length: 1.0-7.0 cm
  - Petal width: 0.0-3.0 cm
- Watch confidence bar and prediction result

### Explore Metrics
- Check accuracy percentage
- View training loss over time
- See prediction confidence trends
- Browse recent predictions table

## Common Use Cases

### Demo for Presentation
1. Initialize brain
2. Quick train (shows all 15 examples learned)
3. Make several predictions
4. Show metrics dashboard with charts

### Testing Brain Performance
1. Initialize brain
2. Train on specific classes
3. Test predictions with known labels
4. Monitor accuracy metric

### Experimenting with Training
1. Initialize brain
2. Train manually with varying confidence levels
3. Observe how loss changes
4. Test generalization with new examples

## Production Deployment

For deploying to production server, see:
- `README.md` - Full deployment guide
- `backend/README.md` - Backend configuration
- `frontend/README.md` - Frontend build process

## Getting Help

- **Full Documentation**: See `README.md` in this directory
- **API Reference**: See `backend/README.md`
- **Component Guide**: See `frontend/README.md`
- **NIMCP Docs**: See `/path/to/nimcp/docs/`

## What's Included

### Backend (Flask + NIMCP)
- ✅ Brain initialization and management
- ✅ Training endpoints (single & batch)
- ✅ Prediction endpoint with confidence
- ✅ Real-time metrics tracking
- ✅ Iris dataset included

### Frontend (React + Chart.js)
- ✅ Training panel with quick-train
- ✅ Prediction panel with confidence visualization
- ✅ Metrics dashboard with 3 charts
- ✅ Real-time updates (2-second polling)
- ✅ Responsive design

### Features Demonstrated
- ✅ NIMCP brain creation
- ✅ Supervised learning (classification)
- ✅ Real-time inference
- ✅ Metrics tracking (loss, accuracy, timing)
- ✅ Data visualization
- ✅ REST API design
- ✅ Modern web architecture

## Performance

Expected performance on modern hardware:
- **Training**: 1-5ms per example
- **Prediction**: 0.5-2ms per inference
- **Frontend updates**: 60 FPS
- **API latency**: 5-10ms

Enjoy exploring NIMCP! 🧠✨
