# NIMCP 2.5 Service Level Agreement (SLA)

## Service Levels

### Availability
- **Target**: 99.9% uptime (43.2 minutes downtime per month)
- **Measurement**: HTTP health check responses
- **Monitoring**: Prometheus + Grafana alerts

### Performance
| Metric | Target | Measurement |
|--------|--------|-------------|
| Inference Latency (P95) | < 10ms | Per-request timing |
| Inference Latency (P99) | < 50ms | Per-request timing |
| Brain Creation | < 100ms | Initialization timing |
| Memory Usage | < 4GB | Process RSS |
| CPU Usage | < 80% avg | Process CPU % |

### Reliability
- **Error Rate**: < 0.1% of requests
- **Data Durability**: 99.999% (5 nines)
- **Backup Frequency**: Every 6 hours
- **Recovery Time Objective (RTO)**: < 1 hour
- **Recovery Point Objective (RPO)**: < 6 hours

## Incident Response

### Severity Levels
1. **Critical (P1)**: Service completely down - Response: 15 min, Resolution: 1 hour
2. **High (P2)**: Major functionality impaired - Response: 30 min, Resolution: 4 hours
3. **Medium (P3)**: Minor issues - Response: 2 hours, Resolution: 24 hours
4. **Low (P4)**: Cosmetic/non-urgent - Response: 1 day, Resolution: 1 week

### Escalation Path
1. On-call engineer (Slack alert)
2. Team lead (15 min)
3. Engineering manager (30 min)
4. CTO (1 hour for P1)

## Monitoring & Alerting

### Critical Alerts
- Service health check fails (3 consecutive)
- Memory usage > 90%
- CPU usage > 95% for 5 minutes
- Error rate > 1%
- Inference latency P99 > 100ms

### Warning Alerts
- Memory usage > 75%
- CPU usage > 80%
- Disk usage > 80%
- Inference latency P95 > 20ms

## Maintenance Windows
- **Scheduled**: Every Sunday 02:00-04:00 UTC
- **Emergency**: As needed with 1-hour notice
- **Notification**: Email + Slack 24 hours before

## Support
- **Hours**: 24/7 for P1, Business hours for P2-P4
- **Channels**:
  - Email: support@nimcp.io
  - Slack: #nimcp-support
  - Phone: +1-XXX-XXX-XXXX (P1 only)

## Compensation
- < 99.9%: 10% monthly credit
- < 99%: 25% monthly credit
- < 95%: 50% monthly credit
