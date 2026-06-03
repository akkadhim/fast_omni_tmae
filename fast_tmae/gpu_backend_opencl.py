"""
OpenCL GPU Backend for TMAE Training
Supports Intel Iris Xe, AMD GPUs, and other devices that support OpenCL
"""

import numpy as np
import pathlib
from typing import Tuple, Optional
import warnings

try:
    import pyopencl as cl
    PYOPENCL_AVAILABLE = True
except ImportError:
    PYOPENCL_AVAILABLE = False
    warnings.warn("PyOpenCL not installed. GPU acceleration with OpenCL will not be available. "
                  "Install with: pip install pyopencl")


class OpenCLBackend:
    """GPU acceleration backend using OpenCL for cross-platform GPU support"""
    
    def __init__(self):
        self.ctx = None
        self.queue = None
        self.program = None
        self.device = None
        self.device_name = None
        self.kernels = {}  # Cache for kernel instances
        self._initialize_opencl()
    
    def _initialize_opencl(self):
        """Initialize OpenCL context and device"""
        if not PYOPENCL_AVAILABLE:
            raise RuntimeError("PyOpenCL is not installed. Install with: pip install pyopencl")
        
        try:
            # Get platforms
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found on this system")
            
            # Prefer Intel, then AMD, then NVIDIA, then others
            device = self._select_best_device(platforms)
            
            if device is None:
                raise RuntimeError("No suitable OpenCL device found")
            
            self.device = device
            self.device_name = device.name
            self.ctx = cl.Context([device])
            self.queue = cl.CommandQueue(self.ctx)
            
            print(f"[OpenCL] Using device: {device.name}")
            print(f"[OpenCL] Device type: {cl.device_type.to_string(device.type)}")
            print(f"[OpenCL] Max compute units: {device.max_compute_units}")
            print(f"[OpenCL] Max work group size: {device.max_work_group_size}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCL: {e}")
    
    def _select_best_device(self, platforms):
        """Select the best available GPU device"""
        device_priority = {
            'Intel': 3,
            'AMD': 2,
            'NVIDIA': 1,
        }
        
        best_device = None
        best_priority = -1
        
        for platform in platforms:
            devices = platform.get_devices(cl.device_type.GPU)
            
            for device in devices:
                priority = device_priority.get(platform.name.split()[0], 0)
                if priority > best_priority:
                    best_priority = priority
                    best_device = device
        
        # If no GPU found, try CPU
        if best_device is None:
            for platform in platforms:
                devices = platform.get_devices(cl.device_type.CPU)
                if devices:
                    best_device = devices[0]
                    break
        
        return best_device
    
    def compile_kernels(self, kernel_source_path: str):
        """Compile OpenCL kernels from source"""
        try:
            with open(kernel_source_path, 'r') as f:
                source = f.read()
            
            self.program = cl.Program(self.ctx, source).build()
            
            # Cache kernel instances to avoid repeated retrieval overhead
            self.kernels['reset_ta_state'] = cl.Kernel(self.program, 'reset_ta_state_masked')
            self.kernels['train_epoch'] = cl.Kernel(self.program, 'tmae_combined_example_and_update')
            self.kernels['collect_embedding'] = cl.Kernel(self.program, 'tmae_collect_embedding')
            
            print(f"[OpenCL] Kernels compiled successfully")
            
        except cl.Error as e:
            raise RuntimeError(f"OpenCL compilation error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to compile kernels: {e}")
    
    def reset_ta_state(self, ta_state: np.ndarray, number_of_state_bits: int):
        """Reset TA state on GPU"""
        # Only allocate if not already done
        if not hasattr(self, '_reset_buf'):
            self._reset_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, 
                                       hostbuf=ta_state)
        else:
            cl.enqueue_copy(self.queue, self._reset_buf, ta_state)
        
        total_elements = ta_state.size
        self.kernels['reset_ta_state'](
            self.queue,
            (256,),  # global work size
            (256,),  # local work size
            self._reset_buf,
            np.uint32(total_elements),
            np.int32(number_of_state_bits)
        )
        
        cl.enqueue_copy(self.queue, ta_state, self._reset_buf)
        self.queue.finish()
    
    def train_epoch(self,
                   ta_state: np.ndarray,
                   number_of_clauses: int,
                   number_of_literals: int,
                   number_of_ta_chunks: int,
                   number_of_state_bits: int,
                   class_weights: np.ndarray,
                   Y: int,
                   T: int,
                   s: float,
                   global_seed: int,
                   X_csr_indptr: np.ndarray,
                   X_csr_indices: np.ndarray,
                   number_of_rows: int,
                   X_csc_indptr: np.ndarray,
                   X_csc_indices: np.ndarray,
                   number_of_cols: int,
                   classes: np.ndarray,
                   target: int,
                   accumulation: int,
                   example_seed: int):
        """Execute one training epoch on GPU (reuses GPU buffers)"""
        
        try:
            # Only allocate buffers on first call - store them as instance variables
            if not hasattr(self, '_gpu_buffers'):
                self._allocate_gpu_buffers(ta_state, class_weights, X_csr_indptr, X_csr_indices, 
                                          X_csc_indptr, X_csc_indices, classes)
            
            # Update only the data that changed
            cl.enqueue_copy(self.queue, self._gpu_buffers['ta_state'], ta_state)
            cl.enqueue_copy(self.queue, self._gpu_buffers['class_weights'], class_weights)
            
            # Calculate shared memory size
            number_of_features = number_of_cols
            number_of_literals_val = 2 * number_of_features
            number_of_literal_chunks = (number_of_literals_val - 1) // 32 + 1
            shared_mem_size = number_of_literal_chunks * np.uint32(0).itemsize
            
            # Launch kernel using cached kernel instance
            threads_per_block = 64
            global_work_size = (((number_of_clauses + threads_per_block - 1) // threads_per_block) * threads_per_block,)
            local_work_size = (threads_per_block,)
            
            self.kernels['train_epoch'](
                self.queue,
                global_work_size,
                local_work_size,
                self._gpu_buffers['ta_state'],
                np.int32(number_of_clauses),
                np.int32(number_of_literals),
                np.int32(number_of_ta_chunks),
                np.int32(number_of_state_bits),
                self._gpu_buffers['class_weights'],
                np.uint32(Y),
                np.int32(T),
                np.float32(s),
                np.uint32(global_seed),
                self._gpu_buffers['indptr_row'],
                self._gpu_buffers['indices_row'],
                np.int32(number_of_rows),
                self._gpu_buffers['indptr_col'],
                self._gpu_buffers['indices_col'],
                np.int32(number_of_cols),
                self._gpu_buffers['classes'],
                np.int32(target),
                np.int32(accumulation),
                np.uint32(example_seed),
                cl.LocalMemory(shared_mem_size)
            )
            
            # Copy result back
            cl.enqueue_copy(self.queue, ta_state, self._gpu_buffers['ta_state'])
            self.queue.finish()
            
        except Exception as e:
            raise RuntimeError(f"OpenCL training epoch failed: {e}")
    
    def _allocate_gpu_buffers(self, ta_state, class_weights, X_csr_indptr, X_csr_indices, 
                              X_csc_indptr, X_csc_indices, classes):
        """Allocate GPU buffers once and reuse them"""
        self._gpu_buffers = {
            'ta_state': cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=ta_state),
            'class_weights': cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=class_weights),
            'indptr_row': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_csr_indptr),
            'indices_row': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_csr_indices),
            'indptr_col': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_csc_indptr),
            'indices_col': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_csc_indices),
            'classes': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=classes),
        }
    
    def collect_embeddings(self,
                          ta_state: np.ndarray,
                          number_of_clauses: int,
                          number_of_literals: int,
                          number_of_ta_chunks: int,
                          number_of_state_bits: int,
                          class_weights: np.ndarray,
                          number_of_features: int) -> np.ndarray:
        """Collect embeddings from GPU (reuses allocations)"""
        
        try:
            # Reuse or allocate embedding buffers
            if not hasattr(self, '_embedding_bufs'):
                self._embedding_bufs = {
                    'ta_state': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=ta_state),
                    'class_weights': cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=class_weights),
                    'sums': cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=number_of_features * np.dtype(np.int32).itemsize),
                    'counts': cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, size=number_of_features * np.dtype(np.uint32).itemsize),
                }
            else:
                # Update data in existing buffers
                cl.enqueue_copy(self.queue, self._embedding_bufs['ta_state'], ta_state)
                cl.enqueue_copy(self.queue, self._embedding_bufs['class_weights'], class_weights)
            
            # Initialize buffers using memset pattern
            zero_int32 = np.int32(0)
            zero_uint32 = np.uint32(0)
            cl.enqueue_fill_buffer(self.queue, self._embedding_bufs['sums'], zero_int32, 
                                  0, number_of_features * np.dtype(np.int32).itemsize)
            cl.enqueue_fill_buffer(self.queue, self._embedding_bufs['counts'], zero_uint32,
                                  0, number_of_features * np.dtype(np.uint32).itemsize)
            
            # Launch kernel using cached kernel instance
            threads_per_block = 256
            global_work_size = (((number_of_literals + threads_per_block - 1) // threads_per_block) * threads_per_block,)
            local_work_size = (threads_per_block,)
            
            self.kernels['collect_embedding'](
                self.queue,
                global_work_size,
                local_work_size,
                self._embedding_bufs['ta_state'],
                np.int32(number_of_clauses),
                np.int32(number_of_literals),
                np.int32(number_of_ta_chunks),
                np.int32(number_of_state_bits),
                self._embedding_bufs['class_weights'],
                np.int32(number_of_features),
                self._embedding_bufs['sums'],
                self._embedding_bufs['counts']
            )
            
            # Copy results back
            embedding_sums = np.zeros(number_of_features, dtype=np.int32)
            embedding_counts = np.zeros(number_of_features, dtype=np.uint32)
            cl.enqueue_copy(self.queue, embedding_sums, self._embedding_bufs['sums'])
            cl.enqueue_copy(self.queue, embedding_counts, self._embedding_bufs['counts'])
            self.queue.finish()
            
            # Calculate final embeddings
            embeddings = np.zeros(number_of_features, dtype=np.uint32)
            for i in range(number_of_features):
                if embedding_counts[i] > 0:
                    embeddings[i] = embedding_sums[i] // embedding_counts[i]
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"OpenCL embedding collection failed: {e}")
    
    def cleanup(self):
        """Clean up OpenCL resources"""
        if hasattr(self, '_gpu_buffers'):
            for key, buf in self._gpu_buffers.items():
                buf.release()
            self._gpu_buffers.clear()
        
        if self.queue:
            self.queue.finish()
        if self.ctx:
            self.ctx = None
