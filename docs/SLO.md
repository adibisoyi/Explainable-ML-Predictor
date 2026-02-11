# Production SLOs and Load-Test Policy

## SLO definitions

- **Availability:** `/health` and `/predict` should have 99.9% monthly success rate.
- **Latency:** `/predict` p95 latency <= 250ms and p99 <= 500ms under baseline load.
- **Correctness guardrail:** online predicted probability distribution should not deviate from training baseline by more than configured drift threshold for >15 minutes.
- **Error budget:** at most 43.2 minutes of unavailable API time per 30-day window.

## Load-test scenarios

1. **Baseline**: 20 concurrent users for 10 minutes, mixed `/predict` and `/health`.
2. **Spike**: ramp from 20 to 100 users over 2 minutes, hold for 5 minutes.
3. **Soak**: 10 users for 6 hours to detect memory or latency creep.

## Execute load tests

```bash
locust -f tests/load/locustfile.py --host http://127.0.0.1:8000
```

The PR/Release checklist should include attached locust HTML reports and percentile tables.
