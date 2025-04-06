#!/bin/sh

echo "Waiting for ollama service..."

timeout=60
while ! curl -s "$OLLAMA_HOST" >/dev/null; do
    sleep 1
    ((timeout--))
    if [[ $timeout -le 0 ]]; then
        echo "Error: Ollama server did not start in time!"
        exit 1
    fi
done


echo "Ollama is running, starting the chatbot..."

python app.py
