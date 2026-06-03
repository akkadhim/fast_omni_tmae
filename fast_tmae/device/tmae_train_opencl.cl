// OpenCL kernel implementation for TMAE training
// Works on Intel Iris Xe, AMD, NVIDIA (non-CUDA), and other GPUs

// ============================================================================
// Helper Functions (Device-side)
// ============================================================================

inline uint calculate_filter(int number_of_literals) {
    uint rem = number_of_literals % 32;
    if (rem != 0) {
        return ~(0xffffffffu << rem);
    } else {
        return 0xffffffffu;
    }
}

inline uint calculate_ta_chunks(int number_of_literals) {
    return (number_of_literals - 1) / 32 + 1;
}

// Simple per-work-item xorshift32 RNG
inline uint xorshift32_next(uint *state) {
    uint x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

inline bool should_update(uint *rng_state, float update_p) {
    if (update_p <= 0.0f) return false;
    if (update_p >= 1.0f) return true;
    uint r = xorshift32_next(rng_state);
    ulong limit = (ulong)((float)0xFFFFFFFFu * update_p);
    return (ulong)r <= limit;
}

inline void cb_inc(__global uint *ta_state, uint active, int number_of_state_bits) {
    uint carry = active;
    for (int b = 0; b < number_of_state_bits; ++b) {
        uint new_carry = ta_state[b] & carry;
        ta_state[b] ^= carry;
        carry = new_carry;
    }
    uint mask = (carry > 0) ? ~0u : 0u;
    for (int b = 0; b < number_of_state_bits; ++b) {
        ta_state[b] |= (mask & carry);
    }
}

inline void cb_dec(__global uint *ta_state, uint active, int number_of_state_bits) {
    uint carry = active;
    for (int b = 0; b < number_of_state_bits; ++b) {
        uint ta_val = ta_state[b];
        uint new_carry = (~ta_val) & carry;
        ta_state[b] = ta_val ^ carry;
        carry = new_carry;
    }
    uint mask = (carry > 0) ? ~0u : 0u;
    for (int b = 0; b < number_of_state_bits; ++b) {
        ta_state[b] &= ~(mask & carry);
    }
}

inline uint calculate_clause_output_update(
    const __global uint *ta_state,
    int number_of_ta_chunks,
    int number_of_state_bits,
    uint filter,
    const __local uint *Xi)
{
    int state_offset = number_of_state_bits - 1;
    uint mismatch = 0;
    for (int k = 0; k < number_of_ta_chunks; ++k) {
        uint pos = k * number_of_state_bits + state_offset;
        uint ta_state_val = (k == (number_of_ta_chunks - 1)) ? (ta_state[pos] & filter) : ta_state[pos];
        mismatch |= (ta_state_val & Xi[k]) ^ ta_state_val;
    }
    return (mismatch == 0) ? 1u : 0u;
}

inline int number_of_include_actions(
    const __global uint *ta_state,
    int clause,
    int number_of_literals,
    int number_of_state_bits,
    int number_of_ta_chunks)
{
    size_t clause_pos = (size_t)clause * (size_t)number_of_ta_chunks * (size_t)number_of_state_bits;
    int state_offset = number_of_state_bits - 1;
    int count = 0;
    
    for (int k = 0; k < (int)number_of_ta_chunks - 1; ++k) {
        uint ta_pos = k * number_of_state_bits + state_offset;
        uint val = ta_state[clause_pos + ta_pos];
        count += popcount(val);
    }
    
    uint last_ta_pos = (number_of_ta_chunks - 1) * number_of_state_bits + state_offset;
    uint filter = calculate_filter(number_of_literals);
    count += popcount(ta_state[clause_pos + last_ta_pos] & filter);
    
    return count;
}

// ============================================================================
// KERNEL 1: Reset TA State
// ============================================================================

__kernel void reset_ta_state_masked(
    __global uint* ta_state,
    uint total_elements,
    int number_of_state_bits)
{
    uint idx = get_global_id(0);
    if (idx >= total_elements) return;

    int k_index = idx % number_of_state_bits;

    if (k_index == number_of_state_bits - 1) {
        ta_state[idx] = 0u; 
    } else {
        ta_state[idx] = ~0u;
    }
}

// ============================================================================
// KERNEL 2: Combined Example Generation and Clause Update
// ============================================================================

__kernel void tmae_combined_example_and_update(
    __global uint *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks,
    int number_of_state_bits,
    __global const int *class_weights,
    uint Y,
    int T,
    float s,
    uint global_seed,
    __global const uint *d_indptr_row,
    __global const uint *d_indices_row,
    int number_of_rows,
    __global const uint *d_indptr_col,
    __global const uint *d_indices_col,
    int number_of_cols,
    __global const uint *d_classes,
    int target,
    int accumulation,
    uint example_seed,
    __local uint *shared_X)
{
    int number_of_features = number_of_cols;
    int number_of_literals_val = 2 * number_of_features;
    uint number_of_literal_chunks = (number_of_literals_val - 1) / 32 + 1;
    
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint gid = get_global_id(0);
    
    // Initialize shared memory X
    for (uint chunk = lid; chunk < number_of_literal_chunks; chunk += lsize) {
        shared_X[chunk] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Set negative literals to 1
    uint neg_start_chunk = number_of_features / 32;
    uint neg_start_pos = number_of_features % 32;
    
    for (uint chunk = neg_start_chunk + (neg_start_pos ? 1 : 0) + lid;
         chunk < number_of_literal_chunks && chunk * 32 < number_of_literals_val;
         chunk += lsize) {
        uint bits_in_chunk = (chunk * 32 + 32 <= number_of_literals_val) ? 32 :
                            (number_of_literals_val - chunk * 32);
        shared_X[chunk] = (bits_in_chunk == 32) ? 0xFFFFFFFFU : ((1U << bits_in_chunk) - 1);
    }
    
    if (lid == 0 && neg_start_pos > 0) {
        uint mask = 0xFFFFFFFFU << neg_start_pos;
        uint bits_in_chunk = (neg_start_chunk * 32 + 32 <= number_of_literals_val) ?
                            32 : (number_of_literals_val - neg_start_chunk * 32);
        uint end_mask = (bits_in_chunk < 32) ? ((1U << bits_in_chunk) - 1) : 0xFFFFFFFFU;
        atomic_or(&shared_X[neg_start_chunk], (mask & end_mask));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Generate example in parallel
    uint target_class = d_classes[target];
    uint target_start = d_indptr_col[target_class];
    uint target_end   = d_indptr_col[target_class + 1];
    uint target_size = target_end - target_start;
    
    uint rng_state = example_seed ^ (lid * 0xdeadbeef);
    
    for (int a = lid; a < accumulation; a += lsize) {
        uint r = xorshift32_next(&rng_state);
        int row;
        
        if (Y) {
            if (target_size > 0) {
                uint random_offset = r % target_size;
                row = (int)d_indices_col[target_start + random_offset];
            } else {
                row = (int)(r % number_of_rows);
            }
        } else {
            row = (int)(r % number_of_rows);
        }
        
        for (uint k = d_indptr_row[row]; k < d_indptr_row[row + 1]; ++k) {
            int feature_idx = (int)d_indices_row[k];
            
            int chunk_nr = feature_idx / 32;
            int chunk_pos = feature_idx % 32;
            int neg_chunk_nr = (feature_idx + number_of_features) / 32;
            int neg_chunk_pos = (feature_idx + number_of_features) % 32;
            
            uint mask = 1u << chunk_pos;
            uint negmask = ~(1u << neg_chunk_pos);
            
            atomic_or(&shared_X[chunk_nr], mask);
            atomic_and(&shared_X[neg_chunk_nr], negmask);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Update clauses - ONE WORK ITEM PER CLAUSE
    int clause_idx = get_global_id(0);
    if (clause_idx >= number_of_clauses) return;
    
    uint clause_rng = global_seed ^ (clause_idx * 0x84c23451);
    
    size_t clause_pos = (size_t)clause_idx * (size_t)number_of_ta_chunks * (size_t)number_of_state_bits;
    __global uint *clause_ta_state = &ta_state[clause_pos];
    
    uint filter = calculate_filter(number_of_literals);
    
    uint clause_output = calculate_clause_output_update(
        clause_ta_state,
        number_of_ta_chunks,
        number_of_state_bits,
        filter,
        shared_X
    );
    
    int Wi = class_weights[clause_idx];
    int class_sum = Wi * (int)clause_output;
    float update_p = 0.0f;
    
    uint use_sparse_feedback = (s > 1.0f) ? 1u : 0u;
    uint max_included_literals = 3;
    
    uint feedback_mask = 0xffffffffu;
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
            for (int i = 0; i < active; ++i) {
                uint r = xorshift32_next(&clause_rng);
                uint pos = r % 32u;
                feedback_mask |= (1u << pos);
            }
        }
    }
    
    if (Y == 1) {
        update_p = ((float)(T - class_sum)) / (2.0f * (float)T);
        if (Wi < 0 || !should_update(&clause_rng, update_p)) {
            return;
        }
        
        int num_includes = number_of_include_actions(ta_state, clause_idx, number_of_literals, 
                                                      number_of_state_bits, number_of_ta_chunks);
        
        if (clause_output && num_includes <= max_included_literals) {
            uint number_of_literal_chunks_local = (number_of_literals - 1) / 32 + 1;
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                __global uint *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                
                uint xi_val = (k < number_of_literal_chunks_local) ? shared_X[k] : 0;
                
                cb_inc(chunk_ta_state, xi_val, number_of_state_bits);
                
                uint excluded = ~xi_val;
                if (use_sparse_feedback) {
                    cb_dec(chunk_ta_state, excluded & feedback_mask, number_of_state_bits);
                } else {
                    cb_dec(chunk_ta_state, excluded, number_of_state_bits);
                }
            }
        } else {
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                __global uint *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                if (use_sparse_feedback) {
                    cb_dec(chunk_ta_state, feedback_mask, number_of_state_bits);
                } else {
                    cb_dec(chunk_ta_state, 0xffffffffu, number_of_state_bits);
                }
            }
        }
    } else {
        update_p = ((float)(T + class_sum)) / (2.0f * (float)T);
        if (Wi < 0 || !should_update(&clause_rng, update_p)) {
            return;
        }
        
        if (clause_output) {
            for (int k = 0; k < number_of_ta_chunks; ++k) {
                __global uint *chunk_ta_state = clause_ta_state + (size_t)k * (size_t)number_of_state_bits;
                uint xi_val = (k < number_of_literal_chunks) ? shared_X[k] : 0;
                uint notXi = ~xi_val;
                cb_inc(chunk_ta_state, notXi, number_of_state_bits);
            }
        }
    }
}

// ============================================================================
// KERNEL 3: Collect Embeddings
// ============================================================================

__kernel void tmae_collect_embedding(
    __global const uint *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks,
    int number_of_state_bits,
    __global const int *class_weights,
    int number_of_features,
    __global int *embedding_sums,
    __global uint *embedding_counts)
{
    int literal_idx = get_global_id(0);
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
        
        uint state = 0;
        for (int b = 0; b < number_of_state_bits; b++) {
            if (ta_state[clause_pos + b] & (1u << chunk_pos))
                state |= (1u << b);
        }
        
        if (is_positive)
            atomic_add(&embedding_sums[feature_idx], (int)state);
        else
            atomic_add(&embedding_sums[feature_idx], -(int)state);
        
        atomic_inc(&embedding_counts[feature_idx]);
    }
}
