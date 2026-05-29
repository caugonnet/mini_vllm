#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

# start server
python -m mini_vllm.api_server --model_name facebook/opt-125m --max_memory_utilization 0.8 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

# wait for port 8000 to accept connections
for _ in {1..60}; do
  if curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/docs | grep -q 200; then
    break
  fi
  sleep 1
done

# now send request
curl http://localhost:8000/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":64,"ignore_eos":false}'
