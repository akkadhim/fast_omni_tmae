#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <omp.h>
#include <nvml.h>
#include <cstdio>

// --- Visual Guide for Tsetlin Logic ---
// The code implements the clause logic where literals interact with the internal state.
// 

// Thread-safe CUDA error checking for multi-GPU
#define CUDA_CHECK_THREAD(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA_ERROR Thread %d, GPU %d, %s:%d: %s\n", \
                omp_get_thread_num(), device, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        fflush(stderr); \
        goto thread_cleanup; \
    } \
} while(0)

// Helper for kernel launches (cannot use goto inside expression)
#define CUDA_CHECK_KERNEL(call) do { \
    (call); \
    cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "KERNEL_ERROR Thread %d, GPU %d, %s:%d: %s\n", \
                omp_get_thread_num(), device, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        fflush(stderr); \
        goto thread_cleanup; \
    } \
} while(0)

// --- Device Helper Functions ---

__device__ inline unsigned int device_cb_calculate_filter(int number_of_literals) {
    unsigned int rem = number_of_literals % 32;
    if (rem != 0) {
        return ~(0xffffffffu << rem);
    } else {
        return 0xffffffffu;
    }
}

__device__ inline unsigned int device_cb_calculate_ta_chunks(int number_of_literals) {
    return (number_of_literals - 1) / 32 + 1;
}

// Simple per-thread xorshift32 RNG
__device__ inline uint32_t xorshift32_next(uint32_t &state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Host side RNG
uint32_t host_xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

__device__ inline bool device_cb_should_update(uint32_t &rng_state, float update_p) {
    if (update_p <= 0.0f) return false;
    if (update_p >= 1.0f) return true;
    uint32_t r = xorshift32_next(rng_state);
    // Use proper casting to avoid overflow issues
    uint64_t limit = (uint64_t)((double)0xFFFFFFFFu * (double)update_p);
    return (uint64_t)r <= limit;
}

__device__ inline void device_cb_inc(unsigned int *ta_state, unsigned int active, int number_of_state_bits) {
    unsigned int carry = active;
    for (int b = 0; b < number_of_state_bits; ++b) {
        unsigned int new_carry = ta_state[b] & carry;
        ta_state[b] ^= carry;
        carry = new_carry;
    }
    unsigned int mask = (carry > 0) ? ~0u : 0u;
    for (int b = 0; b < number_of_state_bits; ++b) {
        ta_state[b] |= (mask & carry);
    }
}

__device__ inline void device_cb_dec(unsigned int *ta_state, unsigned int active, int number_of_state_bits) {
    unsigned int carry = active;
    for (int b = 0; b < number_of_state_bits; ++b) {
        unsigned int ta_val = ta_state[b];
        unsigned int new_carry = (~ta_val) & carry;
        ta_state[b] = ta_val ^ carry;
        carry = new_carry;
    }
    unsigned int mask = (carry > 0) ? ~0u : 0u;
    for (int b = 0; b < number_of_state_bits; ++b) {
        ta_state[b] &= ~(mask & carry);
    }
}

__device__ inline unsigned int device_cb_calculate_clause_output_update(
    const unsigned int *ta_state,
    int number_of_ta_chunks,
    int number_of_state_bits,
    unsigned int filter,
    const unsigned int *Xi)
{
    const int state_offset = number_of_state_bits - 1;
    unsigned int mismatch = 0;
    for (int k = 0; k < number_of_ta_chunks; ++k) {
        const unsigned int pos = k * number_of_state_bits + state_offset;
        // Apply filter only to the last chunk to ignore padding bits
        const unsigned int ta_state_val = (k == (number_of_ta_chunks - 1)) ? (ta_state[pos] & filter) : ta_state[pos];
        
        // Logic: if Included (1) and Input is 0 -> Mismatch
        mismatch |= (ta_state_val & Xi[k]) ^ ta_state_val;
    }
    return (mismatch == 0) ? 1u : 0u;
}

__device__ inline int device_cb_number_of_include_actions(
    const unsigned int *ta_state,
    int clause,
    int number_of_literals,
    int number_of_state_bits)
{
    unsigned int number_of_ta_chunks = device_cb_calculate_ta_chunks(number_of_literals);
    size_t clause_pos = (size_t)clause * (size_t)number_of_ta_chunks * (size_t)number_of_state_bits;
    int state_offset = number_of_state_bits - 1;
    int number_of_include_actions = 0;
    
    for (int k = 0; k < (int)number_of_ta_chunks - 1; ++k) {
        unsigned int ta_pos = k * number_of_state_bits + state_offset;
        number_of_include_actions += __popc(ta_state[clause_pos + ta_pos]);
    }
    
    unsigned int last_ta_pos = (number_of_ta_chunks - 1) * number_of_state_bits + state_offset;
    unsigned int filter = device_cb_calculate_filter(number_of_literals);
    number_of_include_actions += __popc(ta_state[clause_pos + last_ta_pos] & filter);
    
    return number_of_include_actions;
}

// --- Kernels ---

// Reset TA state kernel
__global__ void reset_ta_state_masked_kernel(
    unsigned int* ta_state,
    size_t total_elements,
    int number_of_state_bits)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int k_index = idx % number_of_state_bits;

    // Initialize state: High bits 0, Low bits 1 (Specific to TMAE initialization logic)
    if (k_index == number_of_state_bits - 1) {
        ta_state[idx] = 0u; 
    } else {
        ta_state[idx] = ~0u;
    }
}

// COMBINED kernel: Generate example + Update clauses
// Each thread handles ONE clause, but they cooperate to generate shared_X first
__global__ void tmae_combined_example_and_update_kernel(
    unsigned int *__restrict__ ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks,
    int number_of_state_bits,
    const int *__restrict__ class_weights,
    unsigned int Y,
    int T,
    float s,
    unsigned int global_seed,
    const unsigned int *__restrict__ d_indptr_row,
    const unsigned int *__restrict__ d_indices_row,
    int number_of_rows,
    const unsigned int *__restrict__ d_indptr_col,
    const unsigned int *__restrict__ d_indices_col,
    int number_of_cols,
    const unsigned int *__restrict__    d_classes,
    int target,
    int accumulation,
    unsigned int example_seed)
{
    // Shared memory for X (example literals)
    // Size should be enough for number_of_literal_chunks
    extern __shared__ unsigned int shared_X[];
    
    const int number_of_features = number_of_cols;
    const int number_of_literals_val = 2 * number_of_features;
    const unsigned int number_of_literal_chunks = (number_of_literals_val - 1) / 32 + 1;
    
    // --- Step 1: Initialize X in parallel (all threads participate) ---
    // Zero out shared memory first to ensure clean state
    for (int chunk = threadIdx.x; chunk < number_of_literal_chunks; chunk += blockDim.x) {
        shared_X[chunk] = 0;
    }
    __syncwarp();
    
    // Set negative literals to 1 (initially included)
    const int neg_start_chunk = number_of_features / 32;
    const int neg_start_pos = number_of_features % 32;
    
    for (int chunk = neg_start_chunk + (neg_start_pos ? 1 : 0) + threadIdx.x;
         chunk < number_of_literal_chunks && chunk * 32 < number_of_literals_val;
         chunk += blockDim.x) {
        const int bits_in_chunk = (chunk * 32 + 32 <= number_of_literals_val) ? 32 :
                                 (number_of_literals_val - chunk * 32);
        shared_X[chunk] = (bits_in_chunk == 32) ? 0xFFFFFFFFU : ((1U << bits_in_chunk) - 1);
    }
    
    // Handle the split chunk where positive ends and negative begins
    if (threadIdx.x == 0 && neg_start_pos > 0) {
        const unsigned int mask = 0xFFFFFFFFU << neg_start_pos;
        const int bits_in_chunk = (neg_start_chunk * 32 + 32 <= number_of_literals_val) ?
                                 32 : (number_of_literals_val - neg_start_chunk * 32);
        const unsigned int end_mask = (bits_in_chunk < 32) ? ((1U << bits_in_chunk) - 1) : 0xFFFFFFFFU;
        // Use atomicOr here just in case, though thread 0 is unique
        atomicOr(&shared_X[neg_start_chunk], (mask & end_mask));
    }
    __syncwarp();
    
    // --- Step 2: Generate example in parallel ---
    unsigned int target_start = __ldg(&d_indptr_col[d_classes[target]]);
    unsigned int target_end   = __ldg(&d_indptr_col[d_classes[target] + 1]);

    unsigned int target_size = target_end - target_start;
    
    uint32_t rng_state = __funnelshift_l(example_seed, threadIdx.x, 16);
    
    // All threads participate in accumulation loop
    for (int a = threadIdx.x; a < accumulation; a += blockDim.x) {
        uint32_t r = xorshift32_next(rng_state);
        int row;
        
        if (Y) { // If target is 1 (Positive example)
            if (target_size > 0) {
                unsigned int random_offset = r % target_size;
                row = (int)d_indices_col[target_start + random_offset];
            } else {
                row = (int)(r % number_of_rows);
            }
        } else { // If target is 0 (Negative/Noise example)
            row = (int)(r % number_of_rows);
        }
        
        // Set features from this row in X
        // CSR traversal: Iterate non-zero columns for this row
        for (unsigned int k = d_indptr_row[row]; k < d_indptr_row[row + 1]; ++k) {
            int feature_idx = (int)d_indices_row[k];
            
            
            // Set Positive Literal
            const int chunk_nr = feature_idx / 32;
            const int chunk_pos = feature_idx % 32;
            
            // Clear Negative Literal (because feature is present)
            const int neg_chunk_nr = (feature_idx + number_of_features) / 32;
            const int neg_chunk_pos = (feature_idx + number_of_features) % 32;
            
            unsigned int mask = 1u << chunk_pos;
            unsigned int negmask = ~(1u << neg_chunk_pos);

            unsigned int old = shared_X[chunk_nr];
            unsigned int newv = old | mask;
            shared_X[chunk_nr] = newv;

            old = shared_X[neg_chunk_nr];
            newv = old & negmask;
            shared_X[neg_chunk_nr] = newv;
            
            // atomicOr(&shared_X[chunk_nr], 1U << chunk_pos);
            
            // atomicAnd(&shared_X[neg_chunk_nr], ~(1U << neg_chunk_pos));
        }
    }
    __syncthreads();
    
    // --- Step 3: Update clauses - ONE THREAD PER CLAUSE ---
    int clause_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (clause_idx >= number_of_clauses) return;
    
    // Initialize per-clause RNG
    uint32_t clause_rng = __funnelshift_l(global_seed, clause_idx, 16);
    
    size_t clause_pos = (size_t)clause_idx * (size_t)number_of_ta_chunks * (size_t)number_of_state_bits;
    unsigned int *clause_ta_state = &ta_state[clause_pos];
    
    unsigned int filter = device_cb_calculate_filter(number_of_literals);
    
    // Calculate clause output
    // Note: This function reads shared_X. Ensure shared_X has valid padding if number_of_ta_chunks > literal_chunks
    unsigned int clause_output = device_cb_calculate_clause_output_update(
        clause_ta_state,
        number_of_ta_chunks,
        number_of_state_bits,
        filter,
        shared_X
    );
    
    int Wi = __ldg(&class_weights[clause_idx]);
    int class_sum = Wi * (int)clause_output;
    float update_p = 0.0f;
    
    const unsigned int use_sparse_feedback = (s > 1.0f) ? 1u : 0u;
    const unsigned int max_included_literals = 3;
    
    unsigned int feedback_mask = 0xffffffffu;
    if (use_sparse_feedback) {
        float p = 1.0f / s;
        int n = number_of_literals;
        int active = (int)(p * n + 0.5f);
        if (active < 0) active = 0;
        if (active > n) active = n;
        
        if (active >= 32) {
            feedback_mask = 0xffffffffu;
        } else {
            feedback_mask = 0;
            // Generate sparse mask
            for (int i = 0; i < active; ++i) {
                uint32_t r = xorshift32_next(clause_rng);
                unsigned int pos = r % 32u;
                feedback_mask |= (1u << pos);
            }
        }
    }
    
    if (Y == 1) {
        // Type Ia/Ib Feedback
        update_p = ((float)(T - class_sum)) / (2.0f * (float)T);
        if (Wi < 0 || !device_cb_should_update(clause_rng, update_p)) {
            return;
        }
        
        int num_includes = device_cb_number_of_include_actions(ta_state, clause_idx, number_of_literals, number_of_state_bits);
        
        if (clause_output && num_includes <= max_included_literals) {
            // Type Ia Feedback: Reinforce True Positive
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                unsigned int *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                
                // Safety check for Shared Memory access
                unsigned int xi_val = (k < number_of_literal_chunks) ? shared_X[k] : 0;
                
                device_cb_inc(chunk_ta_state, xi_val, number_of_state_bits);
                
                unsigned int excluded = ~xi_val;
                if (use_sparse_feedback) {
                    device_cb_dec(chunk_ta_state, excluded & feedback_mask, number_of_state_bits);
                } else {
                    device_cb_dec(chunk_ta_state, excluded, number_of_state_bits);
                }
            }
        } else {
            // Type Ib Feedback: Reduce False Negative
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                unsigned int *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                if (use_sparse_feedback) {
                    device_cb_dec(chunk_ta_state, feedback_mask, number_of_state_bits);
                } else {
                    device_cb_dec(chunk_ta_state, 0xffffffffu, number_of_state_bits);
                }
            }
        }
    } else {
        // Type II Feedback: Suppress False Positive
        update_p = ((float)(T + class_sum)) / (2.0f * (float)T);
        if (Wi < 0 || !device_cb_should_update(clause_rng, update_p)) {
            return;
        }
        
        if (clause_output) {
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                unsigned int *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                unsigned int xi_val = (k < number_of_literal_chunks) ? shared_X[k] : 0;
                unsigned int notXi = ~xi_val;
                device_cb_inc(chunk_ta_state, notXi, number_of_state_bits);
            }
        }
    }
}

// Embedding collection kernel
__global__ void tmae_collect_embedding_kernel(
    const unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks,
    int number_of_state_bits,
    const int *class_weights,
    int number_of_features,
    int *embedding_sums,
    unsigned int *embedding_counts)
{
    int literal_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (literal_idx >= number_of_literals) return;
    
    int feature_idx = literal_idx % number_of_features;
    bool is_positive = (literal_idx < number_of_features);
    
    int ta_chunk = literal_idx / 32;
    int chunk_pos = literal_idx % 32;
    
    for (int clause = 0; clause < number_of_clauses; clause++) {
        if (class_weights[clause] <= 0) continue;
        
        size_t clause_pos =
            (size_t)clause * number_of_ta_chunks * number_of_state_bits +
            (size_t)ta_chunk * number_of_state_bits;
        
        unsigned int state = 0;
        for (int b = 0; b < number_of_state_bits; b++) {
            if (ta_state[clause_pos + b] & (1u << chunk_pos))
                state |= (1u << b);
        }
        
        // Accumulate state values as importance/embedding
        if (is_positive)
            atomicAdd(&embedding_sums[feature_idx], (int)state);
        else
            atomicAdd(&embedding_sums[feature_idx], -(int)state);
        
        atomicAdd(&embedding_counts[feature_idx], 1u);
    }
}

// --- Host Functionality ---
bool is_gpu_available(int device_id) {
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t utilization;
    unsigned int threshold = 80; // Example threshold of 20% utilization

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return false; // Consider less utilization when initialization fails
    }

    // Get the handle for the given device ID
    result = nvmlDeviceGetHandleByIndex(device_id, &device);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get handle for device %d: %s\n", device_id, nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }

    // Get the utilization rates
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "Failed to get utilization rates for device %d: %s\n", device_id, nvmlErrorString(result));
        nvmlShutdown();
        return false;
    }

    // Check if utilization is below the threshold
    if (utilization.gpu < threshold) {
        fprintf(stderr, "GPU %d utilization (%u%%) is below threshold (%u%%)\n", device_id, utilization.gpu, threshold);
        nvmlShutdown();
        return true;
    } else {
        fprintf(stderr, "GPU %d utilization (%u%%) is above threshold (%u%%)\n", device_id, utilization.gpu, threshold);
        nvmlShutdown();
        return false;
    }
}

extern "C"
void tmae_train_cuda(
    int number_of_epochs,
    int number_of_examples,
    const unsigned int *classes,
    int number_of_classes,
    const unsigned int *indptr_row,
    const unsigned int *indices_row,
    int number_of_rows,
    unsigned int *indptr_col,
    unsigned int *indices_col,
    int number_of_cols,
    int accumulation,
    int *class_weights,
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks,
    int number_of_state_bits,
    unsigned int *clause_output,
    int T,
    float s,
    unsigned int *embeddings)
{
    // Sanity checks
    if (number_of_clauses <= 0 || number_of_literals <= 0 || number_of_ta_chunks <= 0 || number_of_state_bits <= 0) {
        fprintf(stderr, "Error: Invalid dimensions passed to tmae_train_cuda\n");
        return;
    }

    int device_count;
    cudaGetDeviceCount(&device_count); 
    if (device_count == 0) {
        fprintf(stderr, "Error: No CUDA devices found.\n");
        return;
    }
    
    // Check available devices
    int *available_devices = (int*)malloc(device_count * sizeof(int));
    int num_available = 0;
    for (int i = 0; i < device_count; i++) {
        if (is_gpu_available(i)) {
            available_devices[num_available++] = i;
        }
    }
    
    if (num_available == 0) {
        fprintf(stderr, "ERROR: No available GPUs found!\n");
        free(available_devices);
        return;
    }
    
    fprintf(stderr, "Using %d available GPUs for training\n", num_available);
    unsigned int global_seed = (unsigned int)time(NULL);
    
    #pragma omp parallel num_threads(num_available)
    {
        // --- 1. DECLARATION ZONE (Must be before any CUDA_CHECK_THREAD) ---
        int thread_id = omp_get_thread_num();
        int device = available_devices[thread_id];
        
        // Integers that were causing the error
        int start_class = 0;
        int end_class = 0;
        uint32_t rng_state = 0;
        
        // Pointers
        unsigned int *d_ta_state = NULL;
        int *d_class_weights = NULL;
        unsigned int *d_indptr_row = NULL;
        unsigned int *d_indices_row = NULL;
        unsigned int *d_indptr_col = NULL;
        unsigned int *d_indices_col = NULL;
        unsigned int *d_classes = NULL;
        
        // Helpers
        size_t ta_state_elems = (size_t)number_of_clauses * (size_t)number_of_ta_chunks * (size_t)number_of_state_bits;
        size_t indices_row_size = indptr_row[number_of_rows];
        size_t indices_col_size = indptr_col[number_of_cols];
        
        // Initialize logic variables
        start_class = (thread_id * number_of_classes) / num_available;
        end_class = ((thread_id + 1) * number_of_classes) / num_available;
        rng_state = global_seed ^ (thread_id * 0x12345678);

        // --- 2. START CUDA OPERATIONS ---
        
        // Set device for this thread (This is where the GOTO might happen)
        CUDA_CHECK_THREAD(cudaSetDevice(device));
        
        fprintf(stderr, "Thread %d using GPU %d for classes %d-%d\n", thread_id, device, start_class, end_class - 1);
        
        // --- 3. ALLOCATIONS ---
        
        // TA State
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_ta_state, ta_state_elems * sizeof(unsigned int)));
        
        // Class Weights
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_class_weights, (size_t)number_of_clauses * sizeof(int)));
        
        // CSR Matrix Structures
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_indptr_row, (size_t)(number_of_rows + 1) * sizeof(unsigned int)));
        CUDA_CHECK_THREAD(cudaMemcpy(d_indptr_row, indptr_row, (size_t)(number_of_rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_indices_row, indices_row_size * sizeof(unsigned int)));
        CUDA_CHECK_THREAD(cudaMemcpy(d_indices_row, indices_row, indices_row_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_indptr_col, (size_t)(number_of_cols + 1) * sizeof(unsigned int)));
        CUDA_CHECK_THREAD(cudaMemcpy(d_indptr_col, indptr_col, (size_t)(number_of_cols + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_indices_col, indices_col_size * sizeof(unsigned int)));
        CUDA_CHECK_THREAD(cudaMemcpy(d_indices_col, indices_col, indices_col_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        // Classes
        CUDA_CHECK_THREAD(cudaMalloc((void**)&d_classes, (size_t)number_of_classes * sizeof(unsigned int)));
        CUDA_CHECK_THREAD(cudaMemcpy(d_classes, classes, (size_t)number_of_classes * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        // --- 4. CLASS LOOP ---
        for (int target = start_class; target < end_class; ++target) {
            
            // Copy weights specific to this target class
            size_t weight_offset = (size_t)target * (size_t)number_of_clauses;
            CUDA_CHECK_THREAD(cudaMemcpy(d_class_weights, &class_weights[weight_offset], (size_t)number_of_clauses * sizeof(int), cudaMemcpyHostToDevice));
            
            // Reset TA State
            int reset_threads = 256;
            size_t reset_blocks = (ta_state_elems + reset_threads - 1) / reset_threads;
            if (reset_blocks > 2147483647) reset_blocks = 2147483647; 
            
            CUDA_CHECK_KERNEL((reset_ta_state_masked_kernel<<< (unsigned int)reset_blocks, reset_threads >>>(
                d_ta_state, ta_state_elems, number_of_state_bits)));

            // Training Loop
            unsigned int epochs = number_of_epochs * number_of_examples;
            for (int epoch = 0; epoch < epochs; ++epoch) {
                host_xorshift32(&rng_state);
                int target_value = rng_state & 1;
                
                int threads_per_block = 64;
                int blocks = (number_of_clauses + threads_per_block - 1) / threads_per_block;
                
                int literal_chunks = (2 * number_of_cols - 1) / 32 + 1;
                size_t shared_mem_size = (size_t)literal_chunks * sizeof(unsigned int);

                // accumulation = 14;
                
                CUDA_CHECK_KERNEL((tmae_combined_example_and_update_kernel<<<blocks, threads_per_block, shared_mem_size>>>(
                    d_ta_state,
                    number_of_clauses,
                    number_of_literals,
                    number_of_ta_chunks,
                    number_of_state_bits,
                    d_class_weights,
                    (unsigned int)target_value,
                    T,
                    s,
                    rng_state,
                    d_indptr_row,
                    d_indices_row,
                    number_of_rows,
                    d_indptr_col,
                    d_indices_col,
                    number_of_cols,
                    d_classes,
                    target,
                    accumulation,
                    rng_state ^ epoch
                )));
                
            }
            host_xorshift32(&rng_state);
            // Sync per epoch to manage command queue
            CUDA_CHECK_THREAD(cudaDeviceSynchronize());
            
            #pragma omp critical
            {
                fprintf(stderr, "\rGPU %d: Class %d finished.", device, target);
                fflush(stderr);
            }
            
            // Collect Embeddings
            int *d_embedding_sums = NULL;
            unsigned int *d_embedding_counts = NULL;
            
            CUDA_CHECK_THREAD(cudaMalloc((void**)&d_embedding_sums, (size_t)number_of_cols * sizeof(int)));
            CUDA_CHECK_THREAD(cudaMalloc((void**)&d_embedding_counts, (size_t)number_of_cols * sizeof(unsigned int)));
            CUDA_CHECK_THREAD(cudaMemset(d_embedding_sums, 0, (size_t)number_of_cols * sizeof(int)));
            CUDA_CHECK_THREAD(cudaMemset(d_embedding_counts, 0, (size_t)number_of_cols * sizeof(unsigned int)));
            
            int embed_threads = 256;
            int embed_blocks = (number_of_literals + embed_threads - 1) / embed_threads;
            
            CUDA_CHECK_KERNEL((tmae_collect_embedding_kernel<<<embed_blocks, embed_threads>>>(
                d_ta_state, number_of_clauses, number_of_literals, number_of_ta_chunks, number_of_state_bits,
                d_class_weights, number_of_cols, d_embedding_sums, d_embedding_counts
            )));
            
            CUDA_CHECK_THREAD(cudaDeviceSynchronize());
            
            // Transfer back
            int *h_embedding_sums = (int *)malloc((size_t)number_of_cols * sizeof(int));
            unsigned int *h_embedding_counts = (unsigned int *)malloc((size_t)number_of_cols * sizeof(unsigned int));
            
            CUDA_CHECK_THREAD(cudaMemcpy(h_embedding_sums, d_embedding_sums, (size_t)number_of_cols * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK_THREAD(cudaMemcpy(h_embedding_counts, d_embedding_counts, (size_t)number_of_cols * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            
            size_t embedding_offset = (size_t)target * (size_t)number_of_cols;
            for (int i = 0; i < number_of_cols; ++i) {
                if (h_embedding_counts[i] > 0) {
                    embeddings[embedding_offset + i] = (unsigned int)(h_embedding_sums[i] / (int)h_embedding_counts[i]);
                } else {
                    embeddings[embedding_offset + i] = 0;
                }
            }
            
            free(h_embedding_sums);
            free(h_embedding_counts);
            cudaFree(d_embedding_sums);
            cudaFree(d_embedding_counts);
        }
        
        thread_cleanup:
        if (d_ta_state) cudaFree(d_ta_state);
        if (d_class_weights) cudaFree(d_class_weights);
        if (d_indptr_row) cudaFree(d_indptr_row);
        if (d_indices_row) cudaFree(d_indices_row);
        if (d_indptr_col) cudaFree(d_indptr_col);
        if (d_indices_col) cudaFree(d_indices_col);
        if (d_classes) cudaFree(d_classes);
        
    } // End Parallel
    
    free(available_devices);
    fprintf(stderr, "\nTraining complete.\n");
}