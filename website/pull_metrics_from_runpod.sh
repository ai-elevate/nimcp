#!/bin/bash
# Pull metrics.json from RunPod to Hetzner website dir
scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i /home/bbrelin/.ssh/id_ed25519_runpod -P 16951 root@74.2.96.55:/workspace/nimcp/website/metrics.json /home/bbrelin/nimcp/website/metrics.json 2>/dev/null
