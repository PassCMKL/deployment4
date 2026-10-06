#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Building and starting the MNIST digit recognizer stack..."
docker compose up --build -d

echo "Waiting for the backend to become healthy..."
ready=false
for _ in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null; then
    ready=true
    break
  fi
  sleep 2
done

if [ "$ready" != "true" ]; then
  echo "Backend did not become healthy in time." >&2
  docker compose logs
  exit 1
fi

echo
echo "Stack is running:"
echo "  Frontend: http://localhost:8080"
echo "  Backend:  http://localhost:8000"
echo
echo "Run 'docker compose down' from this directory to stop it."
