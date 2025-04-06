#!/bin/sh

ollama serve > /tmp/server.log 2>&1 & 
SERVER_PID=$!

ollama pull mxbai-embed-large
