This repository contains the work for the paper [FastOmniTMAE: Parallel Clause Learning for Scalable and Hardware-Efficient Tsetlin Embeddings](https://arxiv.org/abs/2605.06982), which presents a new TMAE structure with Omni to build embeddings efficiently.

# 1- Create environment and install Fast TM-AE

## Basic Installation (CPU-Only)
- For Windows
  ```powershell
  python .\scripts\install.py
  ```

- For Linux/MacOS
  ```bash
  python3 scripts/install.py
  ```

## Clean Installation

### To remove old runtime files and folders and start from scratch
- For Windows
  ```powershell
  python .\scripts\install.py --clear
  ```

- For Linux/MacOS
  ```bash
  python3 scripts/install.py --clear
  ```

## GPU Acceleration Options

### For Intel Iris Xe GPUs
```powershell
python .\scripts\install.py --clear --with-opencl
```

### For AMD GPUs
```powershell
python .\scripts\install.py --clear --with-opencl
```

### For NVIDIA GPUs (CUDA)
```powershell
python .\scripts\install.py --clear --with-cuda
```

## Switch to environment
- For Windows
  ```powershell
  .\.venv-custom\Scripts\activate
  ```

- For Linux/MacOS
  ```bash
  source .venv-custom/bin/activate
  ```

---

# 2- Collect embedding for the whole vocabulary

### Download 1 Billion Word corpus to train with big_X.pickle otherwise the model will use IMDb dataset X.pickle
- Download dataset corpus from `1billion` directory in https://drive.google.com/drive/folders/1yUE-wWSvQFQzimdBM4aBJOcjtGgIgUrV?usp=sharing
- Rename `X.pickle` to `big_X.pickle` and place it in `data` directory

### Run and collect embedding for words
- If `--words` is not provided, the script uses the full vocabulary from `vectorizer_X.get_feature_names_out().tolist()`
- If `--words` is provided, the script uses only the passed words
- For Windows
  ```powershell
  python .\scripts\collect_omni.py --results_file results\omni.pickle --device CPU --debug
  ```

- Example with selected words
  ```powershell
  python .\scripts\collect_omni.py --results_file results\omni.pickle --device CPU --words happy sad
  ```

- For Linux/MacOS
  ```bash
  python3 scripts/collect_omni.py --results_file results/omni.pickle --device GPU --debug
  ```

---

# Quick Summary

| GPU Type | Framework | Installation Command |
|----------|-----------|----------------------|
| **Intel Iris Xe** | OpenCL | `python .\scripts\install.py --clear --with-opencl` |
| **AMD Radeon/RDNA** | OpenCL | `python .\scripts\install.py --clear --with-opencl` |
| **NVIDIA GTX/RTX** | CUDA | `python .\scripts\install.py --clear --with-cuda` |
| **Intel Arc GPU** | OpenCL | `python .\scripts\install.py --clear --with-opencl` |
| **No GPU (CPU only)** | - | `python .\scripts\install.py --clear` |
