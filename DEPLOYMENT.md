# NIMCP 2.5 Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Docker Deployment](#docker-deployment)
4. [Systemd Deployment](#systemd-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / Debian 11+ / RHEL 8+
- **CPU**: 2+ cores (4+ recommended)
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk**: 10GB minimum
- **Python**: 3.9+

### Required Software
```bash
# Docker deployment
sudo apt-get install docker.io docker-compose

# Native deployment
sudo apt-get install build-essential cmake python3-dev
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/redmage123/nimcp.git
cd nimcp
```

### 2. Deploy with Docker Compose
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f nimcp
```

### 3. Access Services
- **NIMCP API**: http://localhost:8080
- **Metrics**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

---

## Docker Deployment

### Build Image
```bash
docker build -t nimcp:2.5.0 .
```

### Run Container
```bash
docker run -d \
  --name nimcp \
  -p 8080:8080 \
  -p 9090:9090 \
  -v nimcp-data:/var/lib/nimcp \
  -v nimcp-logs:/var/log/nimcp \
  --restart unless-stopped \
  nimcp:2.5.0
```

### Production Deployment
```bash
# Use deployment script
./deployment/deploy.sh prod

# Or manually with compose
docker-compose -f docker-compose.yml up -d
```

---

## Systemd Deployment

### 1. Build and Install
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/nimcp
make -j$(nproc)
sudo make install
```

### 2. Create User
```bash
sudo useradd -r -s /bin/false nimcp
sudo mkdir -p /var/lib/nimcp /var/log/nimcp
sudo chown nimcp:nimcp /var/lib/nimcp /var/log/nimcp
```

### 3. Install Service
```bash
sudo cp deployment/systemd/nimcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable nimcp
sudo systemctl start nimcp
```

### 4. Check Status
```bash
sudo systemctl status nimcp
sudo journalctl -u nimcp -f
```

---

## Kubernetes Deployment

### 1. Create Namespace
```bash
kubectl create namespace nimcp
```

### 2. Deploy ConfigMap
```bash
kubectl create configmap nimcp-config \
  --from-file=config/nimcp.conf \
  -n nimcp
```

### 3. Deploy Application
```bash
kubectl apply -f deployment/kubernetes/ -n nimcp
```

### 4. Expose Service
```bash
kubectl expose deployment nimcp \
  --type=LoadBalancer \
  --port=8080 \
  -n nimcp
```

---

## Monitoring Setup

### Prometheus
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'nimcp'
    static_configs:
      - targets: ['nimcp:9090']
```

### Grafana Dashboards
1. Access Grafana: http://localhost:3000
2. Add Prometheus data source
3. Import dashboard: `monitoring/grafana/dashboards/nimcp-overview.json`

### Key Metrics
- `nimcp_inference_duration_seconds` - Inference latency
- `nimcp_brain_neurons_total` - Total neurons
- `nimcp_learning_steps_total` - Learning iterations
- `nimcp_memory_usage_bytes` - Memory consumption

---

## Configuration

### Environment Variables
```bash
# Core settings
export NIMCP_HOME=/opt/nimcp
export NIMCP_DATA=/var/lib/nimcp
export NIMCP_LOG=/var/log/nimcp
export NIMCP_LOG_LEVEL=info

# Performance
export NIMCP_MAX_WORKERS=4
export NIMCP_MEMORY_LIMIT=4096

# Features
export NIMCP_ENABLE_METRICS=true
export NIMCP_METRICS_PORT=9090
```

### Configuration File
Edit `/etc/nimcp/nimcp.conf`:
```ini
[core]
log_level = info
max_workers = 4

[brain]
default_size = SMALL
auto_optimize = true

[monitoring]
enable_metrics = true
metrics_port = 9090
```

---

## Troubleshooting

### Check Logs
```bash
# Docker
docker-compose logs nimcp

# Systemd
journalctl -u nimcp -f

# File logs
tail -f /var/log/nimcp/nimcp.log
```

### Health Check
```bash
# HTTP endpoint
curl http://localhost:8080/health

# Container health
docker inspect nimcp | grep -A 5 Health
```

### Common Issues

**Issue**: Container fails to start
```bash
# Check logs
docker logs nimcp

# Verify resources
docker stats nimcp
```

**Issue**: Out of memory
```bash
# Increase memory limit in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 8G
```

**Issue**: Permission denied
```bash
# Fix permissions
sudo chown -R nimcp:nimcp /var/lib/nimcp /var/log/nimcp
```

### Performance Tuning

**Optimize for CPU**:
```bash
# Set CPU affinity
taskset -c 0,1,2,3 /opt/nimcp/bin/integrated_demo
```

**Optimize for Memory**:
```bash
# Enable memory profiling
export NIMCP_ENABLE_MEMORY_PROFILING=true
```

---

## Backup and Recovery

### Backup Data
```bash
# Create backup
docker run --rm \
  -v nimcp-data:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/nimcp-backup.tar.gz /data
```

### Restore Data
```bash
# Restore from backup
docker run --rm \
  -v nimcp-data:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/nimcp-backup.tar.gz -C /
```

---

## Security Best Practices

1. **Run as non-root user** (already configured in Docker/systemd)
2. **Enable TLS** for network communication
3. **Restrict file permissions**: `chmod 600 /etc/nimcp/nimcp.conf`
4. **Regular updates**: `docker pull nimcp:latest`
5. **Monitor logs** for suspicious activity
6. **Backup regularly** (automated recommended)

---

## Production Checklist

- [ ] Resource limits configured
- [ ] Health checks enabled
- [ ] Monitoring configured (Prometheus + Grafana)
- [ ] Log aggregation setup
- [ ] Backup automated
- [ ] Alerts configured
- [ ] Security hardening applied
- [ ] Load testing completed
- [ ] Rollback procedure documented
- [ ] On-call rotation established

---

## Support

- **Documentation**: https://github.com/redmage123/nimcp/docs
- **Issues**: https://github.com/redmage123/nimcp/issues
- **Discussions**: https://github.com/redmage123/nimcp/discussions

---

**Version**: 2.5.0
**Last Updated**: 2025-10-30
