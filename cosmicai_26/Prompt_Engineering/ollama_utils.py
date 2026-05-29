import subprocess
import os
import signal
import time
import atexit
import requests
import json
import time
from pprint import pprint
import threading
import subprocess

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

def _monitor_vram(stop_event, interval=0.2):
    """Monitor VRAM usage with nvidia-smi and return peak in MiB."""
    peak_vram = 0
    while not stop_event.is_set():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            usage_list = [int(v) for v in result.stdout.strip().splitlines() if v.strip()]
            if usage_list:
                peak_vram = max(peak_vram, max(usage_list))
        except Exception:
            pass
        time.sleep(interval)
    return peak_vram


def generate_with_ollama(
    model: str = "gemma3:1b",
    system_prompt: str = "You are a helpful AI assistant.",
    user_prompt: str = "Write a short essay about fish.",
    messages: list = None,
    context_length: int = 2048,
    verbose: bool = False,
    pretty_print: str = "pprint",  # "pprint" or "json"
    seed: int = -1,
    temperature: float = 0.8,
) -> dict:
    """
    Generate text using Ollama's /api/chat endpoint, and measure performance + peak VRAM usage.
    Supports multi-turn conversations via optional 'messages' parameter.
    """
    url = "http://localhost:11434/api/chat"

    # Determine messages for payload
    if messages is not None:
        payload_messages = messages
    else:
        payload_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    payload = {
        "model": model,
        "messages": payload_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": context_length,
            "seed": seed,
        },
    }

    # Start VRAM monitoring in a background thread
    stop_event = threading.Event()
    peak_vram = {"value": 0}

    def monitor():
        peak = _monitor_vram(stop_event)
        peak_vram["value"] = peak

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()

    # Measure wall time
    start_time = time.time()
    response = requests.post(url, json=payload)
    wall_time = time.time() - start_time

    # Stop monitoring
    stop_event.set()
    monitor_thread.join(timeout=1)

    ns_to_s = 1e-9

    if response.status_code == 200:
        result = response.json()
        stats = {
            "response": result["message"]["content"],
            "model": model,
            "context_length": f"{context_length} tokens",
            "total_duration": f"{result.get('total_duration', 0) * ns_to_s:.3f}s",
            "load_duration": f"{result.get('load_duration', 0) * ns_to_s:.3f}s",
            "prompt_eval_count": f"{result.get('prompt_eval_count', 0)} tokens",
            "prompt_eval_duration": f"{result.get('prompt_eval_duration', 0) * ns_to_s:.3f}s",
            "eval_count": f"{result.get('eval_count', 0)} tokens",
            "eval_duration": f"{result.get('eval_duration', 0) * ns_to_s:.3f}s",
            "wall_time": f"{wall_time:.3f}s",
            "tokens_per_second": f"{result.get('eval_count', 0) / max((result.get('eval_duration', 1) * ns_to_s), 0.001):.2f} tokens/s",
            "peak_vram": f"{peak_vram['value']} MiB",
            "seed": seed,
            "temperature": temperature,
        }
    else:
        stats = {
            "error": f"HTTP {response.status_code}: {response.text}",
            "model": model,
            "peak_vram": f"{peak_vram['value']} MiB",
            "seed": seed,
            "temperature": temperature,
        }

    if verbose:
        if pretty_print == "pprint":
            pprint(stats, indent=2, width=80, sort_dicts=False)
        elif pretty_print == "json":
            print(json.dumps(stats, indent=2))

    return stats
