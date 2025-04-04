from fastapi import FastAPI
import subprocess
import threading
import uvicorn

app = FastAPI()

def run_tg_bot():
    subprocess.run(["python", "main.py"])

@app.on_event("startup")
def startup_event():
    threading.Thread(target=run_tg_bot, daemon=True).start()

@app.get("/")
def read_root():
    return {"status": "Telegram bot is running..."}

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=7860)