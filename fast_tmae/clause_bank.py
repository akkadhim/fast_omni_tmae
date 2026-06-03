import numpy as np
from fast_tmae.host.build.fast_tmaelib import ffi, lib
import os
import pathlib

class ClauseBank:
    
    def __init__(
            self,
            seed: int,
            X_shape: tuple,
            s: float,
            T: int,
            number_of_clauses: int,
            max_included_literals: int,
            number_of_state_bits_ta: int,
            batch_size: int,
            output_active: np.ndarray,
            platform="CPU",
            backend="cpu",
            device_ids=None,
            **kwargs
    ):
        assert isinstance(number_of_clauses, int)

        # Initialize random number generator and basic parameters
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.number_of_clauses = int(number_of_clauses)
        
        # Base parameters
        self.s = s
        self.T = T
        
        self.platform = platform
        self.backend = backend  # 'cpu', 'cuda', or 'opencl'
        self.device_ids = device_ids
        
        # Initialize GPU backend flags
        self.is_cuda_gpu = False
        self.is_opencl_gpu = False
        
        if self.platform == "GPU" or self.platform == "CUDA":
            self._initialize_gpu(backend)
        
        # Calculate dimensions - autoencoder only uses 2D data
        if len(X_shape) != 2:
            raise RuntimeError(f"Autoencoder requires 2D input data, got shape: {X_shape}")

        self.number_of_features = X_shape[1]
        self.number_of_literals = self.number_of_features * 2
        self.number_of_ta_chunks = int((self.number_of_literals - 1) / 32 + 1)
        self.max_included_literals = max_included_literals if max_included_literals else self.number_of_literals

        # ClauseBank specific parameters
        assert isinstance(number_of_state_bits_ta, int)
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.batch_size = batch_size

        # Output classes management (replaces SparseClauseContainer)
        self.output_active = output_active
        self.number_of_classes = output_active.shape[0]
        
        # Weight banks for each class (replaces WeightBank instances)
        # introduce positive_percentage to devices the percentage of positive weights
        positive_percentage = 0.5
        self.class_weights = {}
        for class_id in range(self.number_of_classes):
            self.class_weights[class_id] = self.rng.choice([-1, 1], size=self.number_of_clauses, p=[positive_percentage, 1 - positive_percentage]).astype(np.int32)

        # Initialize arrays
        self.clause_output = np.empty(self.number_of_clauses, dtype=np.uint32, order="c")

        self.initialize_clauses()
        self._init_cffi_pointers()

        # Set random seeds
        if self.seed is not None:
            assert isinstance(self.seed, int), "Seed must be a integer"
            native_seed = self.seed
        else:
            native_seed = int(self.rng.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        lib.pcg32_seed(native_seed)
        lib.xorshift128p_seed(native_seed)
        
        self.embeddings = np.zeros(self.number_of_classes * self.number_of_features, dtype=np.uint32)
    
    def _initialize_gpu(self, backend="cuda"):
        """Initialize GPU backend (CUDA or OpenCL)"""
        if backend == "opencl":
            self._initialize_opencl()
        elif backend == "cuda":
            self._initialize_cuda()
        else:
            raise ValueError(f"Unknown GPU backend: {backend}")
    
    def _initialize_opencl(self):
        """Initialize OpenCL GPU backend for Intel Iris Xe / AMD GPUs"""
        try:
            from fast_tmae.gpu_backend_opencl import OpenCLBackend
            import pathlib
            
            self.gpu_backend = OpenCLBackend()
            
            # Load OpenCL kernels
            current_dir = pathlib.Path(__file__).parent
            kernel_path = current_dir / "device" / "tmae_train_opencl.cl"
            
            if not kernel_path.exists():
                raise FileNotFoundError(f"OpenCL kernel file not found: {kernel_path}")
            
            self.gpu_backend.compile_kernels(str(kernel_path))
            self.is_opencl_gpu = True
            self.is_cuda_gpu = False
            
            print(f"[OpenCL] GPU backend initialized successfully: {self.gpu_backend.device_name}")
            
        except ImportError as e:
            raise RuntimeError(f"Failed to initialize OpenCL backend. Make sure PyOpenCL is installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCL GPU: {e}")
            
    def _initialize_cuda(self):
        try:
            import pycuda.driver as cuda
        except Exception as e:
            raise RuntimeError(
                "Could not import pycuda. Install it with 'pip install pycuda' to use the CUDA backend."
            ) from e

        if self.device_ids is not None:
            if not isinstance(self.device_ids, (list, tuple)):
                raise TypeError("device_ids must be a list or tuple of CUDA device ids")
            if len(self.device_ids) == 0:
                raise ValueError("device_ids cannot be empty")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device_id) for device_id in self.device_ids)

        cuda.init()
        self.device = cuda.Device(0)
        self.cuda_ctx = self.device.make_context()
        
        # Load the compiled CUDA library
        import ctypes
        cuda_lib_path = self._get_or_compile_cuda_lib()
        self.cuda_lib = ctypes.CDLL(cuda_lib_path)
        self.tmae_train_cuda_lib = self.cuda_lib.tmae_train_cuda
        
        # Pop context after initialization - we'll push it when needed
        self.cuda_ctx.pop()
        
        self.is_cuda_gpu = True
        self.is_opencl_gpu = False
        print(f"[CUDA] GPU backend initialized successfully on device {self.device.name()}")
    
    def _get_or_compile_cuda_lib(self):
        """Get or compile the CUDA library"""
        import subprocess
        import os
        
        current_dir = pathlib.Path(__file__).parent
        cuda_src = current_dir / "device" / "tmae_train_cuda.cu"
        cuda_build_dir = current_dir / "device" / "build"
        cuda_lib = cuda_build_dir / "libtmae_train.so"
        
        # Create build directory if it doesn't exist
        cuda_build_dir.mkdir(exist_ok=True)
        
        # Compile if not already compiled
        if not cuda_lib.exists():
            print(f"Compiling CUDA library to {cuda_lib}")
            cmd = [
                "nvcc",
                "-shared",
                "-Xcompiler", "-fPIC",
                "-Xcompiler", "-fopenmp",  
                str(cuda_src),
                "-o", str(cuda_lib),
                "-lnvidia-ml" 
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"CUDA compilation failed:\n{result.stderr}")
                raise RuntimeError(f"Failed to compile CUDA library: {result.stderr}")
        
        return str(cuda_lib)

    def _init_cffi_pointers(self):
        """Initialize CFFI pointers for C library calls"""
        self.co_p = ffi.cast("unsigned int *", self.clause_output.ctypes.data)
        # Clause bank pointers
        self.ptr_ta_state = ffi.cast("unsigned int *", self.clause_bank.ctypes.data)

    def initialize_clauses(self):
        """Initialize clause bank arrays"""
        self.clause_bank = np.empty(
            shape=(self.number_of_clauses, self.number_of_ta_chunks, self.number_of_state_bits_ta),
            dtype=np.uint32,
            order="c"
        )

        # Initialize clause bank with default values
        self.clause_bank[:, :, 0: self.number_of_state_bits_ta - 1] = np.array(~0).astype(np.uint32)
        self.clause_bank[:, :, self.number_of_state_bits_ta - 1] = 0
        self.clause_bank = np.ascontiguousarray(self.clause_bank.reshape(
            (self.number_of_clauses * self.number_of_ta_chunks * self.number_of_state_bits_ta)))

    # Weight management methods (replacing WeightBank)
    def get_weights(self, class_id: int) -> np.ndarray:
        """Get weights for a specific class"""
        return self.class_weights[class_id]

    def prepare_X_autoencoder(self, X_csr, X_csc, active_output):
        """Prepare data structures for autoencoder"""
        X = np.ascontiguousarray(np.empty(int(self.number_of_ta_chunks), dtype=np.uint32))
        return X_csr, X_csc, active_output, X

    def produce_autoencoder_example(self, encoded_X, target, target_true_p, accumulation):
        (X_csr, X_csc, active_output, X) = encoded_X
        target_value = self.rng.random() <= target_true_p
        lib.produce_autoencoder_example(
            ffi.cast("unsigned int *", active_output.ctypes.data), 
            active_output.shape[0],
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
            int(X_csr.shape[0]),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
            int(X_csc.shape[1]),
            ffi.cast("unsigned int *", X.ctypes.data),
            int(target),
            int(target_value),
            int(accumulation)
        )

        return X.reshape((1, -1)), target_value
    
    def clause_update(self, encoded_X, target_weights, Y):
        """Calculate clause outputs for updates"""
        xi_p = ffi.cast("unsigned int *", encoded_X[0, :].ctypes.data)
        wi = ffi.cast("int *", target_weights.ctypes.data)
        lib.cb_clause_update(
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_state_bits_ta,
            self.co_p,
            xi_p,
            wi,
            Y.astype(np.uint32),
            self.T,
            self.s
        )

        return self.clause_output
    
    def train_cpu(self, number_of_examples, encoded_X, accumulation):
        (X_csr, X_csc, active_output, X) = encoded_X
        flat_class_weights = np.ascontiguousarray(np.vstack([self.class_weights[c] for c in range(self.number_of_classes)]).astype(np.int32))
        lib.tmae_train(
            int(number_of_examples),
            ffi.cast("unsigned int *", active_output.ctypes.data), 
            active_output.shape[0],
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indptr).ctypes.data),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csr.indices).ctypes.data),
            int(X_csr.shape[0]),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indptr).ctypes.data),
            ffi.cast("unsigned int *", np.ascontiguousarray(X_csc.indices).ctypes.data),
            int(X_csc.shape[1]),
            int(accumulation),
            ffi.cast("int *", flat_class_weights.ctypes.data),
            self.ptr_ta_state,
            self.number_of_clauses,
            self.number_of_literals,
            self.number_of_ta_chunks,
            self.number_of_state_bits_ta,
            self.co_p,
            self.T,
            self.s
        )
    
    def train_gpu(self, number_of_epochs, number_of_examples, encoded_X, accumulation):
        """Train using GPU backend (OpenCL or CUDA)"""
        (X_csr, X_csc, active_output, X) = encoded_X
        flat_class_weights = np.ascontiguousarray(np.vstack([self.class_weights[c] for c in range(self.number_of_classes)]).astype(np.int32))
        
        if self.is_opencl_gpu:
            self._train_gpu_opencl(number_of_epochs, number_of_examples, X_csr, X_csc, active_output, flat_class_weights, accumulation)
        elif self.is_cuda_gpu:
            self._train_gpu_cuda(number_of_epochs, number_of_examples, X_csr, X_csc, active_output, flat_class_weights, accumulation)
        else:
            raise RuntimeError("GPU training requested but no GPU backend is initialized")
    
    def _train_gpu_opencl(self, number_of_epochs, number_of_examples, X_csr, X_csc, active_output, flat_class_weights, accumulation):
        """Train using OpenCL backend (Intel Iris Xe / AMD GPUs)"""
        import random
        
        print(f"[OpenCL] Starting GPU training: {number_of_epochs} epochs, {number_of_examples} examples per epoch")
        
        try:
            # Allocate GPU buffers once (reuse across all epochs)
            X_csr_indptr = np.ascontiguousarray(X_csr.indptr).astype(np.uint32)
            X_csr_indices = np.ascontiguousarray(X_csr.indices).astype(np.uint32)
            X_csc_indptr = np.ascontiguousarray(X_csc.indptr).astype(np.uint32)
            X_csc_indices = np.ascontiguousarray(X_csc.indices).astype(np.uint32)
            
            # Reset TA state on GPU once
            print("[OpenCL] Resetting TA state...")
            self.gpu_backend.reset_ta_state(self.clause_bank, self.number_of_state_bits_ta)
            
            # Training loop - epochs only (not epochs * examples)
            global_seed = self.seed if self.seed else int(np.random.rand() * 2**32)
            
            for epoch in range(number_of_epochs):
                for example_id in range(number_of_examples):
                    # Random target class and Y value
                    target = random.randint(0, self.number_of_classes - 1)
                    Y = random.randint(0, 1)
                    example_seed = global_seed ^ (epoch * number_of_examples + example_id)
                    
                    # Execute training example on GPU (reuse buffers)
                    self.gpu_backend.train_epoch(
                        ta_state=self.clause_bank,
                        number_of_clauses=self.number_of_clauses,
                        number_of_literals=self.number_of_literals,
                        number_of_ta_chunks=self.number_of_ta_chunks,
                        number_of_state_bits=self.number_of_state_bits_ta,
                        class_weights=flat_class_weights[target],
                        Y=Y,
                        T=self.T,
                        s=self.s,
                        global_seed=global_seed,
                        X_csr_indptr=X_csr_indptr,
                        X_csr_indices=X_csr_indices,
                        number_of_rows=X_csr.shape[0],
                        X_csc_indptr=X_csc_indptr,
                        X_csc_indices=X_csc_indices,
                        number_of_cols=X_csc.shape[1],
                        classes=active_output,
                        target=target,
                        accumulation=accumulation,
                        example_seed=example_seed
                    )
                
                print(f"[OpenCL] Epoch {epoch + 1}/{number_of_epochs} completed")
            
            # Collect embeddings from GPU
            print("[OpenCL] Collecting embeddings...")
            for target in range(self.number_of_classes):
                embeddings = self.gpu_backend.collect_embeddings(
                    ta_state=self.clause_bank,
                    number_of_clauses=self.number_of_clauses,
                    number_of_literals=self.number_of_literals,
                    number_of_ta_chunks=self.number_of_ta_chunks,
                    number_of_state_bits=self.number_of_state_bits_ta,
                    class_weights=flat_class_weights[target],
                    number_of_features=self.number_of_features
                )
                self.embeddings[target * self.number_of_features:(target + 1) * self.number_of_features] = embeddings
            
            print("[OpenCL] GPU training completed successfully")
            
        except Exception as e:
            print(f"[OpenCL] Error during GPU training: {e}")
            raise
    
    def _train_gpu_cuda(self, number_of_epochs, number_of_examples, X_csr, X_csc, active_output, flat_class_weights, accumulation):
        """Train using CUDA backend (NVIDIA GPUs)"""
        print(f"[CUDA] Starting GPU training: {number_of_epochs} epochs, {number_of_examples} examples per epoch")
        
        try:
            self.cuda_ctx.push()
            import ctypes
            from ctypes import c_int, c_float, POINTER, c_uint32
            self.tmae_train_cuda_lib.argtypes = [
                c_int,  # number_of_epochs
                c_int,  # number_of_examples
                POINTER(c_uint32),  # classes
                c_int,  # number_of_classes
                POINTER(c_uint32),  # indptr_row
                POINTER(c_uint32),  # indices_row
                c_int,  # number_of_rows
                POINTER(c_uint32),  # indptr_col
                POINTER(c_uint32),  # indices_col
                c_int,  # number_of_cols
                c_int,  # accumulation
                POINTER(ctypes.c_int32),  # class_weights
                POINTER(c_uint32),  # ta_state
                c_int,  # number_of_clauses
                c_int,  # number_of_literals
                c_int,  # number_of_ta_chunks
                c_int,  # number_of_state_bits
                POINTER(c_uint32),  # clause_output
                c_int,  # T
                c_float,  # s
                POINTER(c_uint32),  # embeddings
            ]
            
            self.tmae_train_cuda_lib.restype = None
            # Call the CUDA function
            self.tmae_train_cuda_lib(
                np.int32(number_of_epochs),
                np.int32(number_of_examples),
                active_output.ctypes.data_as(POINTER(c_uint32)),
                np.int32(active_output.shape[0]),
                np.ascontiguousarray(X_csr.indptr).ctypes.data_as(POINTER(c_uint32)),
                np.ascontiguousarray(X_csr.indices).ctypes.data_as(POINTER(c_uint32)),
                np.int32(X_csr.shape[0]),
                np.ascontiguousarray(X_csc.indptr).ctypes.data_as(POINTER(c_uint32)),
                np.ascontiguousarray(X_csc.indices).ctypes.data_as(POINTER(c_uint32)),
                np.int32(X_csc.shape[1]),
                np.int32(accumulation),
                flat_class_weights.ctypes.data_as(POINTER(ctypes.c_int32)),
                self.clause_bank.ctypes.data_as(POINTER(c_uint32)),
                np.int32(self.number_of_clauses),
                np.int32(self.number_of_literals),
                np.int32(self.number_of_ta_chunks),
                np.int32(self.number_of_state_bits_ta),
                self.clause_output.ctypes.data_as(POINTER(c_uint32)),
                np.int32(self.T),
                np.float32(self.s),
                self.embeddings.ctypes.data_as(POINTER(c_uint32))
            )
            
            print("[CUDA] GPU training completed successfully")
            
        except Exception as e:
            print(f"[CUDA] Error during GPU training: {e}")
            raise
        finally:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
    
    def get_ta_state(self, clause, ta):
        """Get the state of a specific TA"""
        ta_chunk = ta // 32
        chunk_pos = ta % 32
        pos = int(clause * self.number_of_ta_chunks * self.number_of_state_bits_ta + 
                  ta_chunk * self.number_of_state_bits_ta)
        
        state = 0
        for b in range(self.number_of_state_bits_ta):
            if self.clause_bank[pos + b] & (1 << chunk_pos) > 0:
                state |= (1 << b)
        return state

    # Serialization support
    def __getstate__(self):
        """Custom pickling support"""
        state = {k: v for k, v in self.__dict__.items() 
                if not k.startswith('ptr_') and not k.endswith('_p') and 
                not k in ['co_p', 'cob_p', 'tiafc_p', 'lcm_p', 'lcmp_p', 'flpc_p', 'previous_xi_p']}
        return state

    def __setstate__(self, state):
        """Custom unpickling support"""
        self.__dict__ = state
        self._init_cffi_pointers()
