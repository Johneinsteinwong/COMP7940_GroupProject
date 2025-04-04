#!/bin/sh

ollama serve > /tmp/server.log 2>&1 & 
sleep 1 # wait for 1s
ollama pull mxbai-embed-large &
pull_pid=$!
wait $pull_pid # wait for pull to finish

python app.py
