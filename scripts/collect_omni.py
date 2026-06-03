import argparse
import os
import pickle
import shutil
import time
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]

# Add the scripts directory to the path so we can import run_omnitm
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(repo_root))

_dll_dirs = []
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    build_dir = repo_root / "fast_tmae" / "host" / "build"
    if build_dir.exists():
        _dll_dirs.append(os.add_dll_directory(str(build_dir)))

    gcc_path = shutil.which("gcc")
    if gcc_path:
        _dll_dirs.append(os.add_dll_directory(str(Path(gcc_path).resolve().parent)))

import run_omnitm as omnitm
from fast_tmae.autoencoder import TMAutoEncoder 

def main(results_file, device, words=None, cpu_threads=None, debug=False):
    data_dir = repo_root / "data"
    x_path = data_dir / "big_X.pickle"
    vectorizer_path = data_dir / "big_vectorizer_X.pickle"
    if not x_path.exists():
        x_path = data_dir / "X.pickle"
        vectorizer_path = data_dir / "vectorizer_X.pickle"

    vectorizer_X = pickle.load(open(vectorizer_path, "rb"))
    if not vectorizer_X:
        raise ValueError("Failed to load vectorizer_X from pickle.")
    X = pickle.load(open(x_path, "rb"))
    if X is None:
        raise ValueError("Failed to load X from pickle.")

    if device.upper() == "CPU" and getattr(X, "format", None) == "csr":
        X = X.reshape(X.shape[0], -1)
        X = X.__class__(
            (np.ones(X.indices.shape[0], dtype=np.uint8), X.indices, X.indptr),
            shape=X.shape,
            copy=False,
        )

    if words is None:
        words = vectorizer_X.get_feature_names_out().tolist()
    print(f"Number of words to process: {len(words)}")
    if not words:
        raise ValueError("No words found in vectorizer_X vocabulary.")
    
    results_path = Path(results_file)
    if not results_path.is_absolute():
        results_path = repo_root / results_path
    results_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    word_vectors = omnitm.build_embedding(
        TMAutoEncoder,
        X,
        vectorizer_X,
        words,
        device.upper(),
        cpu_threads=cpu_threads,
        debug=debug,
    )
    end_time = time.time()
    
    with open(results_path, "wb") as f:
        pickle.dump(word_vectors, f)    
        
    elapsed_time = end_time - start_time
    formatted_elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print(f"Embeddings collected and saved to {results_path}")
    print(f"Elapsed time: {formatted_elapsed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, default="results/omni.pickle")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--words", nargs="+", help="Optional list of words to embed")
    parser.add_argument("--cpu_threads", type=int, help="Optional number of CPU threads to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug")
    args = parser.parse_args()
    if args.debug:
        print("[+] Running in debug mode")
        
    # Clean up any previous CUDA builds to force recompilation
    cuda_path = repo_root / "fast_tmae" / "device" / "build"
    if cuda_path.exists():
        print(f"[+] Removing folder {cuda_path.relative_to(repo_root)}")
        shutil.rmtree(cuda_path, ignore_errors=True)
        
    main(args.results_file, args.device, args.words, args.cpu_threads, args.debug)
