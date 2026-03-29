#!/bin/bash
# Increase proxy timeout for chat API (model loading takes ~30s first call)
sudo sed -i 's/proxy_read_timeout 60s;/proxy_read_timeout 120s;/' /etc/nginx/sites-enabled/nimcp
sudo nginx -t && sudo nginx -s reload && echo "Timeout increased to 120s"
