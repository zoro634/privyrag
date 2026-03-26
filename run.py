import subprocess
import sys
import time
import os
from pathlib import Path

def main():
    print("Starting PrivyRAG Services...")
    
    base_dir = Path(__file__).parent.resolve()
    
    # Enable virtual environment if it exists
    python_exec = sys.executable
    if (base_dir / "venv" / "Scripts" / "python.exe").exists():
        python_exec = str(base_dir / "venv" / "Scripts" / "python.exe")
        
    # 1. Start FastAPI backend
    print(f"Starting Backend API on localhost:8000...")
    backend_process = subprocess.Popen(
        [python_exec, "-m", "uvicorn", "backend.app:app", "--host", "localhost", "--port", "8000"],
        cwd=base_dir
    )
    
    # Give backend a moment to start
    time.sleep(3)
    
    # 2. Start Next.js frontend
    print(f"Starting Next.js Frontend on port 3000...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(base_dir.parent / "privyrag-frontend"),
        shell=True
    )
    
    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
