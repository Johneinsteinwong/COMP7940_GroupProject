#!/bin/sh

ollama serve > server.log 2>&1 & 
sleep 1 
ollama pull mxbai-embed-large &
pull_pid=$!
wait $pull_pid # wait for pull to finish

python main.py
