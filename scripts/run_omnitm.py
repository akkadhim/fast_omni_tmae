from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import multiprocessing
import os
from tempfile import TemporaryDirectory

import tqdm
import numpy as np
from scipy.sparse import csr_matrix

_PROCESS_X_CSR = None
_PROCESS_X_CSC = None
_PROCESS_MEMMAPS = ()


class SparseIndexMatrix:
    def __init__(self, shape, indptr, indices):
        self.shape = tuple(shape)
        self.indptr = indptr
        self.indices = indices


def _build_cpu_embedding_for_word(
    TMAutoEncoder,
    X_train,
    target_word,
    word_id,
    number_of_features,
    number_of_clauses,
    T,
    s,
    accumulation,
    max_included_literals,
    output_balancing,
    epochs,
    number_of_examples,
    debug,
    show_epoch_progress,
):
    single_output_active = np.empty(1, dtype=np.uint32)
    single_output_active[0] = word_id

    tm = TMAutoEncoder(
        number_of_clauses=number_of_clauses,
        T=T,
        s=s,
        output_active=single_output_active,
        max_included_literals=max_included_literals,
        accumulation=accumulation,
        feature_negation=True,
        platform='CPU',
        output_balancing=output_balancing
    )
    epoch_iter = range(epochs)
    if show_epoch_progress:
        epoch_iter = tqdm.tqdm(
            epoch_iter,
            desc=f"Training TM for word '{target_word}'",
            unit="epoch",
            leave=False,
        )
    for _ in epoch_iter:
        tm.fit(X_train, number_of_examples=number_of_examples)
    clauses_weights = tm.get_weights(0)

    literal_sums = np.zeros(number_of_features)
    literal_counts = np.zeros(number_of_features)
    if debug:
        plot_clauses(tm)
    
    for j in range(number_of_clauses):
        clause_weight = clauses_weights[j]
        if clause_weight > 0:
            for i in range(tm.clause_bank.number_of_literals):
                if i < number_of_features:
                    literal_sums[i] += tm.get_ta_state(j, i)
                    literal_counts[i] += 1
                else:
                    literal_sums[i - number_of_features] -= tm.get_ta_state(j, i)
                    literal_counts[i - number_of_features] += 1

    non_zero_counts = literal_counts > 0
    embedding = np.zeros(number_of_features)
    embedding[non_zero_counts] = (literal_sums[non_zero_counts] / literal_counts[non_zero_counts]).astype(int)
    return word_id, embedding


def _prepare_shared_cpu_input(X_train):
    if isinstance(X_train, tuple) and len(X_train) == 2:
        return X_train

    if getattr(X_train, "format", None) == "csr":
        X_csr = X_train.reshape(X_train.shape[0], -1)
    else:
        X_csr = csr_matrix(X_train.reshape(X_train.shape[0], -1))

    X_csc = X_csr.tocsc()
    X_csc.sort_indices()
    return X_csr, X_csc


def _write_memmap_array(temp_dir, name, array):
    path = os.path.join(temp_dir, f"{name}.npy")
    np.save(path, np.ascontiguousarray(array, dtype=np.uint32), allow_pickle=False)
    return path


def _prepare_process_matrix_specs(X_train, temp_dir):
    X_csr, X_csc = _prepare_shared_cpu_input(X_train)
    return {
        "shape": X_csr.shape,
        "csr_indptr_path": _write_memmap_array(temp_dir, "csr_indptr", X_csr.indptr),
        "csr_indices_path": _write_memmap_array(temp_dir, "csr_indices", X_csr.indices),
        "csc_indptr_path": _write_memmap_array(temp_dir, "csc_indptr", X_csc.indptr),
        "csc_indices_path": _write_memmap_array(temp_dir, "csc_indices", X_csc.indices),
    }


def _init_process_worker(matrix_specs):
    global _PROCESS_X_CSR, _PROCESS_X_CSC, _PROCESS_MEMMAPS

    csr_indptr = np.load(matrix_specs["csr_indptr_path"], mmap_mode="r", allow_pickle=False)
    csr_indices = np.load(matrix_specs["csr_indices_path"], mmap_mode="r", allow_pickle=False)
    csc_indptr = np.load(matrix_specs["csc_indptr_path"], mmap_mode="r", allow_pickle=False)
    csc_indices = np.load(matrix_specs["csc_indices_path"], mmap_mode="r", allow_pickle=False)

    _PROCESS_MEMMAPS = (csr_indptr, csr_indices, csc_indptr, csc_indices)
    _PROCESS_X_CSR = SparseIndexMatrix(matrix_specs["shape"], csr_indptr, csr_indices)
    _PROCESS_X_CSC = SparseIndexMatrix(matrix_specs["shape"], csc_indptr, csc_indices)


def _chunk_word_list(word_list, batch_size):
    for start_index in range(0, len(word_list), batch_size):
        yield word_list[start_index:start_index + batch_size]


def _build_cpu_embedding_batch(
    word_batch,
    number_of_features,
    number_of_clauses,
    T,
    s,
    accumulation,
    max_included_literals,
    output_balancing,
    epochs,
    number_of_examples,
):
    from fast_tmae.autoencoder import TMAutoEncoder as WorkerTMAutoEncoder

    X_shared = (_PROCESS_X_CSR, _PROCESS_X_CSC)
    return [
        _build_cpu_embedding_for_word(
            WorkerTMAutoEncoder,
            X_shared,
            target_word,
            word_id,
            number_of_features,
            number_of_clauses,
            T,
            s,
            accumulation,
            max_included_literals,
            output_balancing,
            epochs,
            number_of_examples,
            False,
            False,
        )
        for target_word, word_id in word_batch
    ]

def detect_gpu_backend():
    """
    Detect available GPU backend:
    - 'opencl' for Intel Iris Xe / AMD GPUs
    - 'cuda' for NVIDIA GPUs
    - 'cpu' if no GPU backend is available
    """
    backend = None
    
    # Try OpenCL first (Intel Iris Xe, AMD GPUs)
    try:
        import pyopencl as cl
        platforms = cl.get_platforms()
        if platforms:
            backend = 'opencl'
            device_name = platforms[0].get_devices()[0].name if platforms[0].get_devices() else "Unknown"
            print(f"[GPU] OpenCL backend available on: {device_name}")
            return backend
    except ImportError:
        pass
    
    # Try CUDA next (NVIDIA GPUs)
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() > 0:
            backend = 'cuda'
            device = cuda.Device(0)
            print(f"[GPU] CUDA backend available on: {device.name()}")
            return backend
    except ImportError:
        pass
    except Exception:
        pass
    
    print("[GPU] No GPU backend detected, falling back to CPU")
    return 'cpu'

def build_embedding(TMAutoEncoder, X_train, vectorizer_X, words, device, cpu_threads=None, debug=False):
    print(f"Building embeddings for {len(words)} words using device: {device}")
    vocabulary = vectorizer_X.vocabulary_
    number_of_features = len(vocabulary)

    # Hyperparameters for TMAutoEncoder
    number_of_clauses=32
    T=20000
    s=1.0
    accumulation=24
    max_included_literals=3
    output_balancing=0.5
    epochs=4
    number_of_examples=2000
    
    word_list = []
    for word in words:
        if isinstance(word, list):
            print(f"Warning: word '{word}' is a list, skipping.")
            continue
        word_id = vocabulary.get(word)
        if word_id is not None:
            word_list.append((word, word_id))
        else:
            print(f"Word '{word}' not found in vocabulary, skipping.")

    all_embeddings = {}

    if device in {"GPU", "CUDA", "OPENCL"}:
        if device == "OPENCL":
            gpu_backend = "opencl"
        elif device == "CUDA":
            gpu_backend = "cuda"
        else:
            # Auto-detect GPU backend (OpenCL for Intel/AMD, CUDA for NVIDIA)
            gpu_backend = detect_gpu_backend()
        
        if device == "GPU" and gpu_backend == 'cpu':
            print("[!] Requested GPU mode but no GPU backend found. Falling back to CPU mode.")
            device = "CPU"
        else:
            output_active = np.array([word_id for _, word_id in word_list], dtype=np.uint32)
            tm = TMAutoEncoder(
                    number_of_clauses=number_of_clauses,
                    T=T,
                    s=s,
                    output_active=output_active,
                    max_included_literals=max_included_literals,
                    accumulation=accumulation,
                    feature_negation=True,
                    platform="GPU",
                    backend=gpu_backend,  # Use detected backend (opencl or cuda)
                    device_ids=None, # Use [0, 2] forGPU device 0 and 2 for training
                    output_balancing=output_balancing
                )
            tm.fit(X_train, number_of_epochs=epochs, number_of_examples=number_of_examples)
            
            num_classes = len(word_list)
            embeddings_array = tm.get_embeddings().copy()
            embeddings_array = embeddings_array.view(np.int32).reshape(num_classes, number_of_features)
            for idx, (_, word_id) in enumerate(word_list):
                # Each row idx contains the embedding vector for word at index idx
                all_embeddings[word_id] = embeddings_array[idx]
            
            print(f"Total embeddings built: {len(all_embeddings)}")
            return all_embeddings
    
    if device == "CPU":
        if cpu_threads is not None and cpu_threads < 1:
            raise ValueError("cpu_threads must be at least 1")

        max_workers = cpu_threads if cpu_threads is not None else (os.cpu_count() or 1)
        if debug and max_workers > 1:
            print("[CPU] Debug plotting enabled, running with 1 CPU thread to avoid plot file conflicts.")
            max_workers = 1
        max_workers = min(max_workers, len(word_list))

        if max_workers == 1:
            # Build the sparse views once and reuse them for the sequential CPU path.
            X_shared = _prepare_shared_cpu_input(X_train)
            show_epoch_progress = True
            for target_word, word_id in tqdm.tqdm(word_list, desc="Generating embeddings", unit="word"):
                word_id, embedding = _build_cpu_embedding_for_word(
                    TMAutoEncoder,
                    X_shared,
                    target_word,
                    word_id,
                    number_of_features,
                    number_of_clauses,
                    T,
                    s,
                    accumulation,
                    max_included_literals,
                    output_balancing,
                    epochs,
                    number_of_examples,
                    debug,
                    show_epoch_progress,
                )
                all_embeddings[word_id] = embedding
        else:
            batch_size = max(1, len(word_list) // max(1, max_workers * 4))
            batches = list(_chunk_word_list(word_list, batch_size))
            print(f"[CPU] Using {max_workers} worker processes.")
            with TemporaryDirectory(prefix="fast_tmae_sparse_") as temp_dir:
                matrix_specs = _prepare_process_matrix_specs(X_train, temp_dir)
                with ProcessPoolExecutor(
                    max_workers=max_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_init_process_worker,
                    initargs=(matrix_specs,),
                ) as executor:
                    batch_iterator = executor.map(
                        _build_cpu_embedding_batch,
                        batches,
                        repeat(number_of_features),
                        repeat(number_of_clauses),
                        repeat(T),
                        repeat(s),
                        repeat(accumulation),
                        repeat(max_included_literals),
                        repeat(output_balancing),
                        repeat(epochs),
                        repeat(number_of_examples),
                    )
                    progress = tqdm.tqdm(total=len(word_list), desc="Generating embeddings", unit="word")
                    try:
                        for batch_results in batch_iterator:
                            for word_id, embedding in batch_results:
                                all_embeddings[word_id] = embedding
                            progress.update(len(batch_results))
                    finally:
                        progress.close()

    print(f"Total embeddings built: {len(all_embeddings)}")
    return all_embeddings

def plot_clauses(tm):
    import matplotlib.pyplot as plt

    number_of_literals = tm.clause_bank.number_of_literals // 2
    ta_states = np.zeros((number_of_literals, ), dtype=np.int32)
    # plot all clauses in one figure and save it to a file
    # each subplot expected to get one clause with literals on x-axis (around 40k literals) and ta states on y-axis (0 to 2^8=256)
    plt.figure(figsize=(30, 30))
    clauses_weights = tm.get_weights(0)
    positive_clause_indices = [j for j, w in enumerate(clauses_weights) if w > 0]
    for j in range(len(positive_clause_indices)):
        clause_weight = clauses_weights[positive_clause_indices[j]]
        if clause_weight > 0:
            # only literals after number_of_literals position 
            for i in range(number_of_literals):
                ta_states[i] = tm.get_ta_state(positive_clause_indices[j], i + number_of_literals)
            plt.subplot((len(positive_clause_indices) + 3) // 4, 4, j + 1)
            plt.plot(range(number_of_literals), ta_states, marker='o', markersize=2, linestyle='None')
            plt.title(f"Clause {j} TA States")
            plt.xlabel("Literal Index")
            plt.ylabel("TA State")
            plt.ylim(0, 150)
            plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/clauses_ta_states.png")
