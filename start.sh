#!/bin/sh

ollama serve > /tmp/server.log 2>&1 & 
SERVER_PID=$!

timeout=30
while ! curl -s http://localhost:11434 >/dev/null; do
    sleep 1
    ((timeout--))
    if [[ $timeout -le 0 ]]; then
        echo "Error: Ollama server did not start in time!"
        exit 1
    fi
done

#sleep 1 # wait for 1s
ollama pull mxbai-embed-large &
pull_pid=$!
wait $pull_pid # wait for pull to finish

if ! ollama list | grep -q "mxbai-embed-large"; then
    echo "Error: Model failed to load!"
    exit 1
fi

python app.py
