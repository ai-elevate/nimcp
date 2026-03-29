#!/bin/bash
# Fix: add missing closing brace to HTTP server block
echo "}" | sudo tee -a /etc/nginx/sites-enabled/nimcp > /dev/null
sudo nginx -t && sudo nginx -s reload && echo "Nginx fixed — chat API live on /api/"
