import subprocess
import os
import signal
import time
import atexit

# Global handle to the running Ollama process
OLLAMA_PROCESS = None


def start_ollama_server():
    global OLLAMA_PROCESS

    # Create log file
    log_file = "ollama_server.log"

    # Kill any existing ollama processes
    subprocess.run(["pkill", "ollama"], capture_output=True)
    time.sleep(2)

    # Start ollama serve in background
    print("🚀 Starting Ollama server...")
    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setpgrp,  # or start_new_session=True [web:4][web:16]
    )
    OLLAMA_PROCESS = process  # store the Popen object, not just pid
    print(f"📄 Server logs: {log_file}")
    print(f"📍 API endpoint: http://localhost:11434")

    # Wait for server to start
    print("⏳ Waiting 5 seconds for server startup...")
    time.sleep(5)

    if OLLAMA_PROCESS is not None and OLLAMA_PROCESS.poll() is None:
        print(f"✅ Ollama server ready! PID: {OLLAMA_PROCESS.pid}")
    else:
        print("⚠️ Ollama server did not start correctly")

    # Register cleanup function once
    atexit.register(stop_ollama_server)


def stop_ollama_server():
    global OLLAMA_PROCESS
    if OLLAMA_PROCESS is None:
        print("ℹ️ No Ollama server process recorded")
        return

    try:
        # Kill the whole process group so children die too [web:4][web:11]
        os.killpg(os.getpgid(OLLAMA_PROCESS.pid), signal.SIGTERM)
        print("🛑 Ollama server stopped")
    except ProcessLookupError:
        print("ℹ️ Ollama server process already gone")
    except Exception as e:
        print(f"⚠️ Error stopping Ollama server: {e}")
    finally:
        OLLAMA_PROCESS = None
