#!/bin/sh

ollama serve > /tmp/server.log 2>&1 & 
SERVER_PID=$!

echo "ollama served..."

ollama pull mxbai-embed-large

wait $SERVER_PID
