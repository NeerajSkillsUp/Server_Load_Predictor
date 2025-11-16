import subprocess
import time
import os
import signal
import sys

# --- Configuration for all 9 prediction services ---
# Ensure ports here match the ports in your app_*.py files AND the dashboard.
API_PROCESSES = [
    # A2 Services
    {"name": "A2 Hourly (RF)", "file": "A2 files\\app_M2_hourly.py", "port": 5000},
    {"name": "A2 Daily (XGBoost)", "file": "A2 files\\app_M2_daily.py", "port": 5001},
    {"name": "A2 Weekly (RF)", "file": "A2 files\\app_M2_weekly.py", "port": 5002},
    
    # A7 Services
    {"name": "A7 Hourly (Prophet)", "file": "A7 files\\app_M7_hourly.py", "port": 5003},
    {"name": "A7 Daily (XGBoost)", "file": "A7 files\\app_M7_daily.py", "port": 5004},
    {"name": "A7 Weekly (XGBoost)", "file": "A7 files\\app_M7_weekly.py", "port": 5005},
    
    # A9 Services
    {"name": "A9 Hourly (GLM)", "file": "A9 files\\app_M9_hourly.py", "port": 5006},
    {"name": "A9 Daily (XGBoost)", "file": "A9 files\\app_M9_daily.py", "port": 5007},
    {"name": "A9 Weekly (SARIMA)", "file": "A9 files\\app_M9_weekly.py", "port": 5008},
]

# List to hold the Popen objects
running_processes = []

# Function to find the correct Python executable command
def get_python_command():
    """Tries to find the correct command: 'python3' or 'python'"""
    try:
        subprocess.run(['python3', '--version'], check=True, capture_output=True)
        return 'python3'
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(['python', '--version'], check=True, capture_output=True)
            return 'python'
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ FATAL: Could not find 'python' or 'python3' executable.")
            print("Please ensure Python is installed and available in your system's PATH.")
            sys.exit(1)

# Graceful shutdown function
def shutdown_apis(sig, frame):
    print("\nShutting down all API services...")
    for p in running_processes:
        try:
            # Send SIGTERM (graceful shutdown)
            p.terminate() 
        except ProcessLookupError:
            pass # Process already terminated
            
    # Wait for processes to terminate
    for p in running_processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if still running
            print(f"Force killing process {p.pid}...")
            p.kill()
        except ProcessLookupError:
            pass

    print("All services stopped. Exiting.")
    sys.exit(0)

# Register the shutdown signal handler
signal.signal(signal.SIGINT, shutdown_apis)
signal.signal(signal.SIGTERM, shutdown_apis)

# Main execution
def start_apis():
    python_cmd = get_python_command()
    print(f"Using Python command: {python_cmd}")
    print(f"Starting {len(API_PROCESSES)} API services...")
    print("-" * 35)

    for api in API_PROCESSES:
        if not os.path.exists(api['file']):
            print(f"❌ ERROR: File not found: {api['file']}. Skipping.")
            continue
            
        try:
            # Start the Flask app as a new process
            # Use Popen to run non-blocking
            process = subprocess.Popen([python_cmd, api['file']])
            running_processes.append(process)
            print(f"✅ Started {api['name']} ({api['file']}). Process ID: {process.pid}")
        except Exception as e:
            print(f"❌ ERROR starting {api['name']} ({api['file']}): {e}")
            
    print("-" * 35)
    print("\nAll services are starting up!")
    print("KEEP THIS SCRIPT RUNNING. DO NOT CLOSE IT.")
    print("Press Ctrl+C to shut down gracefully.")

    # Keep the main script alive so the child processes don't terminate
    try:
        while True:
            time.sleep(5)
            # Check for unexpected terminations
            for i, p in enumerate(running_processes):
                if p.poll() is not None:
                    api = API_PROCESSES[i]
                    print(f"\n❌ A service (PID {p.pid}, {api['name']}) terminated unexpectedly!")
                    print("Shutting down all services. Check the log for errors in that script.")
                    shutdown_apis(None, None) 
                    return
    except KeyboardInterrupt:
        # This catch is redundant due to the signal handler, but good practice
        shutdown_apis(None, None)
    except Exception as e:
        # Catch unexpected exceptions in the main loop
        print(f"\nMain loop error: {e}")
        shutdown_apis(None, None)

# Execute the function
if __name__ == "__main__":
    start_apis()