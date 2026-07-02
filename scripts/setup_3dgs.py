import os
import subprocess
import sys

def clone_repo():
    repo_url = "https://github.com/graphdeco-inria/gaussian-splatting"
    target_dir = "gaussian-splatting"
    
    if not os.path.exists(target_dir):
        print(f"Cloning repository from {repo_url}...")
        subprocess.run(["git", "clone", "--recursive", repo_url, target_dir], check=True)
        print("Clone successful.")
    else:
        print(f"Directory '{target_dir}' already exists. Skipping clone.")

def install_dependencies():
    print("Syncing environment with uv sync...")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("Environment synced successfully.")
    except Exception as e:
        print(f"Error during uv sync: {e}")

if __name__ == "__main__":
    clone_repo()
    install_dependencies()
    print("Setup complete.")
