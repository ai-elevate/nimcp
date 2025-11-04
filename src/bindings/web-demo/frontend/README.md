# NIMCP Web Demo - Frontend

React application for interactive NIMCP brain visualization with Chart.js.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start
```

App runs on `http://localhost:3000`

## Build

```bash
# Create production build
npm run build

# Output in build/ directory
# Serve with any static file server
```

## Components

### App.js
Main application component managing state and API communication.

**Features**:
- Brain initialization and reset
- Metrics polling (every 2 seconds)
- API integration with Flask backend
- Error handling and loading states

### TrainingPanel.js
Interface for training the brain.

**Features**:
- Manual single-example training
- Quick-train button (entire dataset)
- Load example buttons for each class
- Form validation
- Training status messages

### PredictionPanel.js
Interface for making predictions.

**Features**:
- Input form for flower measurements
- Optional true label for accuracy tracking
- Prediction result display with confidence bar
- Color-coded confidence (green/orange/red)
- Inference timing display

### MetricsDashboard.js
Real-time metrics and charts using Chart.js.

**Features**:
- 6 metric cards (trained, predictions, accuracy, loss, timing)
- Training loss chart (line chart)
- Prediction confidence chart (line chart)
- Class distribution chart (bar chart)
- Recent predictions table

## Dependencies

- **react** 18.2.0 - UI library
- **react-dom** 18.2.0 - React DOM renderer
- **react-scripts** 5.0.1 - Build tooling
- **chart.js** 4.4.0 - Charting library
- **react-chartjs-2** 5.2.0 - React wrapper for Chart.js
- **axios** 1.6.0 - HTTP client

## Configuration

### API URL

Edit `App.js` to change backend URL:

```javascript
const API_URL = 'https://your-backend.com/api';
```

Or use environment variable:

```bash
# .env.local
REACT_APP_API_URL=https://your-backend.com/api
```

```javascript
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
```

### Polling Interval

Change metrics polling interval in `App.js`:

```javascript
useEffect(() => {
  const interval = setInterval(() => {
    if (brainInitialized) {
      fetchMetrics();
    }
  }, 5000);  // Poll every 5 seconds instead of 2
  ...
}, [brainInitialized]);
```

## Styling

All styles in `App.css` using CSS variables:

```css
:root {
  --primary: #667eea;
  --success: #10b981;
  --danger: #ef4444;
  ...
}
```

Change theme colors by modifying CSS variables.

## Chart Configuration

Customize Chart.js options in `MetricsDashboard.js`:

```javascript
const lossChartOptions = {
  responsive: true,
  plugins: {
    title: {
      display: true,
      text: 'Your Custom Title',
      color: '#ffffff'
    }
  },
  scales: {
    y: {
      beginAtZero: true,
      max: 1.0  // Set max y value
    }
  }
};
```

## Development

### File Structure

```
src/
├── App.js              # Main component
├── App.css             # Styles
├── index.js            # Entry point
└── components/
    ├── TrainingPanel.js
    ├── PredictionPanel.js
    └── MetricsDashboard.js
```

### Adding New Component

1. Create file in `src/components/`
2. Import in `App.js`
3. Add to JSX

```javascript
// src/components/MyComponent.js
import React from 'react';

function MyComponent({ data }) {
  return (
    <div className="my-component">
      <h2>My Component</h2>
      {/* Your content */}
    </div>
  );
}

export default MyComponent;
```

```javascript
// App.js
import MyComponent from './components/MyComponent';

// In JSX:
<MyComponent data={someData} />
```

### Adding New Chart

Use Chart.js types: Line, Bar, Pie, Doughnut, Radar, etc.

```javascript
import { Pie } from 'react-chartjs-2';

const data = {
  labels: ['Class A', 'Class B', 'Class C'],
  datasets: [{
    data: [10, 20, 30],
    backgroundColor: ['#ff6384', '#36a2eb', '#ffce56']
  }]
};

<Pie data={data} options={options} />
```

## Responsive Design

Breakpoints in `App.css`:

```css
@media (max-width: 768px) {
  /* Mobile styles */
  .panels-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  /* Small mobile */
}
```

## Performance

### Optimization Tips

1. **Production build**: Always use `npm run build` for production
2. **Code splitting**: React.lazy() for large components
3. **Memoization**: Use React.memo() for expensive renders
4. **Polling**: Adjust interval based on needs (lower = more responsive, higher = less load)

### Production Checklist

- [ ] Run `npm run build`
- [ ] Test production build locally
- [ ] Check bundle size (`npm run build` shows size)
- [ ] Enable compression (gzip) on server
- [ ] Use CDN for static assets
- [ ] Set proper cache headers

## Deployment

### Static Hosting (Netlify, Vercel)

```bash
# Build
npm run build

# Deploy build/ directory
# Netlify/Vercel will do this automatically from Git
```

### Nginx

```nginx
server {
  listen 80;
  server_name example.com;
  root /var/www/nimcp-demo/build;
  index index.html;

  location / {
    try_files $uri $uri/ /index.html;
  }

  # API proxy
  location /api {
    proxy_pass http://localhost:5000;
  }
}
```

### Docker

```dockerfile
FROM node:18 AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Troubleshooting

### "Failed to compile"

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Charts not showing

```bash
# Ensure Chart.js is registered
# Check MetricsDashboard.js has ChartJS.register() call
```

### CORS errors

```bash
# Ensure Flask backend has CORS enabled
# Check backend console for errors
# Verify API_URL is correct
```

### Slow performance

```bash
# Use production build
npm run build

# Check React DevTools Profiler
# Look for unnecessary re-renders
```

## Testing

```bash
# Run tests (if added)
npm test

# Run tests with coverage
npm test -- --coverage

# Run tests in watch mode
npm test -- --watch
```

## Browser Support

- Chrome/Edge: Last 2 versions
- Firefox: Last 2 versions
- Safari: Last 2 versions
- Mobile: iOS Safari 12+, Chrome Android 90+

## Accessibility

- Semantic HTML used throughout
- ARIA labels on interactive elements
- Keyboard navigation support
- Color contrast meets WCAG AA standards

## License

MIT - Same as NIMCP
