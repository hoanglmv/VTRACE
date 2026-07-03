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
    print("Setting up virtual environment...")
    
    if not os.path.exists(".venv"):
        subprocess.run(["uv", "venv", ".venv"], check=True)
    
    print("Installing core dependencies (including PyTorch)...")
    subprocess.run(["uv", "pip", "install", "-e", "."], check=True)
    
    print("Installing simple-knn...")
    subprocess.run(["uv", "pip", "install", "--no-build-isolation", "./gaussian-splatting/submodules/simple-knn"], check=True)
    
    print("Installing diff-gaussian-rasterization (this may take a few minutes to compile CUDA)...")
    subprocess.run(["uv", "pip", "install", "--no-build-isolation", "./gaussian-splatting/submodules/diff-gaussian-rasterization"], check=True)
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    clone_repo()
    install_dependencies()
