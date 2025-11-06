# 🧠 NIMCP Web Demo - Neural Network Interactive Showcase

A comprehensive web-based demonstration of NIMCP's spiking neural network capabilities with real-time visualization, interactive controls, and analytics.

## Features

### Real-Time Visualization
- **Neural Network Topology**: Interactive graph visualization using vis-network
- **Live Node Updates**: Neurons change color based on activity level
- **Connection Display**: Synaptic connections with weight indicators
- **Active Neuron Highlighting**: See neurons firing in real-time

### Analytics Dashboard
- **Network Activity Chart**: Real-time activity level tracking
- **Average Weight Chart**: Monitor synaptic weight changes
- **Spike Events Chart**: Track spike frequency over time
- All charts powered by Chart.js with 100-point history

### Interactive Controls

#### Simulation Control
- **Start/Stop**: Control simulation execution
- **Reset**: Reset network to initial state
- Real-time status indicator

#### Network Modification
- **Add Connections**: Create synapses between any two neurons
- **Set Weights**: Specify connection weights
- **Select Neurons**: Click neurons to view detailed information

#### Plasticity Rules
- **STDP**: Spike-Timing-Dependent Plasticity
- **Oja's Rule**: Hebbian learning with weight normalization
- **Homeostasis**: Maintain target activity levels

#### Network Maintenance
- **Prune Synapses**: Remove weak connections below threshold
- **Weight Statistics**: View weight distribution metrics

### Real-Time Metrics
- Neuron count
- Current timestamp
- Network activity level
- Average synaptic weight

## Quick Start

### Prerequisites
```bash
# Install Flask and Flask-SocketIO
sudo apt install python3-flask python3-flask-socketio

# Or using pip
pip3 install flask flask-socketio
```

### Running the Demo
```bash
cd /home/bbrelin/nimcp/examples/web_demo
python3 app.py
```

The server will start on:
- **Local**: http://localhost:5000
- **Network**: http://0.0.0.0:5000

### Access the Dashboard
Open your web browser and navigate to `http://localhost:5000`

## Architecture

### Backend (Flask)
- **REST API**: 10+ endpoints for network control
- **WebSocket**: Real-time simulation updates via Socket.IO
- **Background Thread**: Simulation loop at 20 steps/second
- **Metrics Collection**: Every 10 simulation steps
- **Homeostasis**: Applied every 100 steps

### Frontend (JavaScript)
- **Chart.js**: Three real-time line charts for metrics
- **vis-network**: Neural network topology visualization
- **Socket.IO Client**: WebSocket communication
- **Responsive UI**: Bootstrap-inspired layout

## API Reference

### REST Endpoints

#### Network Information
```
GET /api/network/info
Returns: { num_neurons, current_time, simulation_running }
```

#### Network Topology
```
GET /api/network/topology
Returns: { nodes: [...], edges: [...] }
```

#### Neuron Details
```
GET /api/neuron/<id>
Returns: { id, state, activity, weight_norm, weight_mean, weight_std, incoming_count }
```

#### Add Connection
```
POST /api/connection/add
Body: { from_id, to_id, weight }
```

#### Simulation Control
```
POST /api/simulation/start
POST /api/simulation/stop
POST /api/simulation/reset
```

#### Apply Plasticity
```
POST /api/plasticity/apply
Body: { rule: "stdp"|"oja"|"homeostasis", neuron_id }
```

#### Prune Network
```
POST /api/network/prune
Body: { threshold }
```

### WebSocket Events

#### Server → Client
- `simulation_update`: { timestamp, active_neurons }
- `metrics_update`: { activity, weight, timestamp }

#### Client → Server
- `connect`: Connection established
- `request_update`: Request current state

## Usage Examples

### Start Simulation
1. Click **"▶️ Start Simulation"** button
2. Watch neurons light up as they fire
3. Observe real-time chart updates

### Add a Connection
1. Enter **From Neuron ID** (e.g., 0)
2. Enter **To Neuron ID** (e.g., 10)
3. Set **Weight** (e.g., 0.5)
4. Click **"➕ Add Synapse"**
5. See the new connection appear in the network visualization

### Apply Plasticity
1. Select plasticity rule from dropdown (STDP, Oja, Homeostasis)
2. Enter target **Neuron ID**
3. Click **"⚡ Apply Plasticity"**
4. Check activity log for results

### Prune Weak Synapses
1. Set **Prune Threshold** (e.g., 0.1)
2. Click **"✂️ Prune Weak Synapses"**
3. Weak connections are removed from the network

### Inspect Neuron
1. Click any neuron in the visualization
2. View detailed information in the panel:
   - ID, State, Activity
   - Weight Norm
   - Incoming Connection Count

## Configuration

### Network Size
Edit `app.py` line 25:
```python
'num_neurons': 50  # Change to desired size
```

### Simulation Speed
Edit `app.py` line 83:
```python
time.sleep(0.05)  # Default: 20 steps/second
```

### Metrics Collection Frequency
Edit `app.py` line 68:
```python
if timestamp % 10 == 0:  # Collect every N steps
```

### Chart History Length
Edit `static/js/app.js` line 14:
```javascript
const chartDataLimit = 100;  // Number of points to display
```

## File Structure

```
examples/web_demo/
├── app.py                  # Flask backend server
├── README.md              # This file
├── templates/
│   └── index.html         # Main dashboard HTML
└── static/
    └── js/
        └── app.js         # Frontend JavaScript
```

## Technical Details

### Network Initialization
- 50 neurons by default
- Initial connections: Each neuron connects to neurons at offsets +5, +7, and +15
- Connection weights: 0.3-0.5 range

### Plasticity Rules
- **STDP**: Updates synapses based on spike timing differences
- **Oja's Rule**: Normalizes weights to prevent runaway growth
- **Homeostasis**: Adjusts thresholds to maintain target activity

### Visualization Colors
- **Blue**: Low activity (< 0.3)
- **Orange**: Medium activity (0.3 - 0.7)
- **Red**: High activity (> 0.7)
- **Green Border**: Currently active/firing

## Performance

- **Simulation**: 20 steps/second
- **Metrics Update**: Every 10 steps (2x/second)
- **WebSocket Latency**: < 50ms typical
- **Chart Rendering**: 60 FPS
- **Network Visualization**: Hardware-accelerated via vis-network

## Troubleshooting

### Port Already in Use
```bash
# Kill existing process
sudo lsof -ti:5000 | xargs kill -9

# Or change port in app.py line 330
```

### WebSocket Warnings
```bash
# Install simple-websocket for better performance
pip3 install simple-websocket
```

### Charts Not Updating
- Check browser console for JavaScript errors
- Verify WebSocket connection is established
- Check that simulation is running

## Future Enhancements

- [ ] Export metrics to CSV/JSON
- [ ] Save/load network configurations
- [ ] Multi-network comparison view
- [ ] GPU acceleration toggle
- [ ] Custom neuron activation functions
- [ ] Advanced topology layouts
- [ ] Recording and playback

## License

Part of the NIMCP (Neural Inference for Massive Concurrent Processing) project.

## Credits

Built with:
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/) - WebSocket support
- [Chart.js](https://www.chartjs.org/) - Real-time charting
- [vis-network](https://visjs.org/) - Network visualization
- [NIMCP](https://github.com/bbrelin/nimcp) - Neural network engine
