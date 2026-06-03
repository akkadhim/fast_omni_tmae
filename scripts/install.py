import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path

def venv_python(venv_root: Path) -> str:
    """Return the path to the python executable inside a venv (cross-platform)."""
    if os.name == "nt":
        return str(venv_root / "Scripts" / "python.exe")
    return str(venv_root / "bin" / "python")

def run(cmd, cwd=None):
    print("[run]", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=cwd)

def main():
    repo_root = Path(__file__).resolve().parents[1]
    venv_custom = repo_root / ".venv-custom"

    parser = argparse.ArgumentParser(description="Prepare virtual environments for custom Fast TM-AE.")
    parser.add_argument("--clear", action="store_true", help="Clear existing virtual environments before creating new ones.")
    parser.add_argument("--with-cuda", action="store_true", help="Install pycuda for NVIDIA GPU support (requires CUDA toolkit installed).")
    parser.add_argument("--with-opencl", action="store_true", help="Install pyopencl for cross-platform GPU support (Intel Iris Xe, AMD, etc).")
    args = parser.parse_args()
    if args.clear:
        print("[+] Clearing existing virtual environments and build artifacts")
        # Remove virtual environments
        if venv_custom.exists():
            print(f"[+] Removing virtual environment {venv_custom}")
            shutil.rmtree(venv_custom, ignore_errors=True)

        # Remove build artifacts
        folders_to_remove = [
            repo_root / "build",
            repo_root / "Release",
            repo_root / "fast_tmae.egg-info",
            repo_root / "fast_tmae" / "__pycache__",
            repo_root / "scripts" / "__pycache__",
            repo_root / "fast_tmae" / "device" / "build",
            repo_root / "fast_tmae" / "host" / "build"
        ]
        for folder_path in folders_to_remove:
            if folder_path.exists():
                print(f"[+] Removing folder {folder_path.relative_to(repo_root)}")
                shutil.rmtree(folder_path, ignore_errors=True)

        files_to_remove = [
            # repo_root / "fast_tmae" / "",
        ]

        # Also remove compiled extension files by pattern per-OS
        # Windows: .pyd; Linux: .so; macOS: .so or .dylib (CFFI wheels typically .so)
        ext_candidates = []
        if os.name == "nt":
            ext_candidates.append(repo_root / "fast_tmae" / "fasttmlib.cp311-win_amd64.pyd")
            # Also any fasttmlib*.pyd
            ext_candidates.extend((repo_root / "fast_tmae").glob("fasttmlib*.pyd"))
        else:
            ext_candidates.extend((repo_root / "fast_tmae").glob("fasttmlib*.so"))
            ext_candidates.extend((repo_root / "fast_tmae").glob("fasttmlib*.dylib"))

        files_to_remove.extend(ext_candidates)

        for file_path in files_to_remove:
            try:
                if Path(file_path).exists():
                    print(f"[+] Removing file {Path(file_path).relative_to(repo_root)}")
                    Path(file_path).unlink(missing_ok=True)
            except TypeError:
                # Python <3.8 compatibility: missing_ok not available
                if Path(file_path).exists():
                    Path(file_path).unlink()

    # Create venvs
    print("[+] Creating virtual environments")
    run([sys.executable, "-m", "venv", str(venv_custom)])
    py_custom = venv_python(venv_custom)

    # Install into custom venv
    print("[+] Installing custom Fast TM-AE into .venv-custom")
    run([py_custom, "-m", "pip", "install", "--upgrade", "pip"]) 
    
    # Install base deps with pinned versions for consistency and pickle compatibility
    base_deps = ["cffi>=1.15.0", "numpy==1.26.4", "scikit-learn==1.6.1", "tqdm", "psutil", "matplotlib"]
    
    if args.with_cuda:
        print("[+] Installing with CUDA support (pycuda)")
        base_deps.append("pycuda")
    elif args.with_opencl:
        print("[+] Installing with OpenCL support (pyopencl) for Intel Iris Xe / AMD GPUs")
        base_deps.append("pyopencl")
    else:
        print("[+] Installing without GPU support (CPU-only mode)")
    
    run([py_custom, "-m", "pip", "install"] + base_deps)

    # Build the native extension in-place so the checkout is directly runnable.
    (repo_root / "fast_tmae" / "host" / "build").mkdir(parents=True, exist_ok=True)
    build_cmd = [py_custom, "setup.py", "build_ext", "--inplace"]
    if os.name == "nt" and shutil.which("gcc"):
        build_cmd.append("--compiler=mingw32")
    run(build_cmd, cwd=repo_root)

    print("[+] Done. Environments prepared.")
    if args.with_cuda:
        print("    GPU acceleration with NVIDIA CUDA enabled.")
    elif args.with_opencl:
        print("    GPU acceleration with OpenCL enabled (Intel Iris Xe, AMD, etc).")
    else:
        print("    Running in CPU-only mode.")
        print("    For Intel Iris Xe GPU support, reinstall with: python .\\scripts\\install.py --clear --with-opencl")
        print("    For NVIDIA GPU support, reinstall with: python .\\scripts\\install.py --clear --with-cuda")

    # Clean build and egg-info from repo
    print("[+] Cleaning build artifacts from repository")
    folders_to_remove = [
        repo_root / "build",
        repo_root / "fast_tmae.egg-info"
    ]
    for folder_path in folders_to_remove:
        if folder_path.exists():
            print(f"[+] Removing folder {folder_path.relative_to(repo_root)}")
            shutil.rmtree(folder_path, ignore_errors=True)

if __name__ == "__main__":
    main()
