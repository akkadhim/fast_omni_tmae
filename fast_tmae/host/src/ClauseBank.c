#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include "fast_rand.h"

#include "ClauseBank.h"
#include "Tools.h"

// Inline helper functions for common operations
static inline unsigned int cb_calculate_filter(int number_of_literals) {
    return (((number_of_literals) % 32) != 0) ? 
           (~(0xffffffff << ((number_of_literals) % 32))) : 0xffffffff;
}

static inline unsigned int cb_calculate_ta_chunks(int number_of_literals) {
    return (number_of_literals - 1) / 32 + 1;
}

static inline int cb_should_update(float update_p) {
    // Avoid expensive float division by comparing integers
    return fast_rand() <= (unsigned int)(FAST_RAND_MAX * update_p);
}

// Input validation macro
#define CB_VALIDATE_INPUTS(condition) \
    do { if (!(condition)) return; } while(0)

#define CB_VALIDATE_INPUTS_INT(condition, default_val) \
    do { if (!(condition)) return (default_val); } while(0)

static inline void cb_initialize_random_streams(unsigned int *feedback_to_ta, int number_of_literals, int number_of_ta_chunks, float s)
{
    CB_VALIDATE_INPUTS(feedback_to_ta && number_of_literals > 0 && number_of_ta_chunks > 0 && s > 0.0f);
    
    // Initialize all bits to zero	
    memset(feedback_to_ta, 0, number_of_ta_chunks * sizeof(unsigned int));

    const int n = number_of_literals;
    const float p = 1.0f / s;

    int active = normal(n * p, n * p * (1.0f - p));
    active = (active >= n) ? n : ((active < 0) ? 0 : active);
    
    // Optimized random selection with reduced collisions
    while (active > 0) {
        const int f = fast_rand() % number_of_literals;
        const int chunk_idx = f / 32;
        const unsigned int bit_mask = 1U << (f % 32);
        
        if (!(feedback_to_ta[chunk_idx] & bit_mask)) {
            feedback_to_ta[chunk_idx] |= bit_mask;
            --active;
        }
    }
}

// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cb_inc(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
{
    unsigned int carry = active;

    for (int b = 0; b < number_of_state_bits; ++b) {
        unsigned int new_carry = ta_state[b] & carry;
        ta_state[b] ^= carry;
        carry = new_carry;
    }

    unsigned int mask = -(carry > 0);
    for (int b = 0; b < number_of_state_bits; ++b)
        ta_state[b] |= mask & carry;
}

// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
static inline void cb_dec(unsigned int *ta_state, unsigned int active, int number_of_state_bits)
{
    unsigned int carry = active;

    for (int b = 0; b < number_of_state_bits; ++b) {
        unsigned int ta_val = ta_state[b];
        unsigned int new_carry = (~ta_val) & carry;  // underflow carry
        ta_state[b] = ta_val ^ carry;                // subtract with XOR
        carry = new_carry;
    }

    // Branchless saturation to 0 for underflows
    unsigned int mask = -(carry > 0);  // all 1s if carry != 0, else 0
    for (int b = 0; b < number_of_state_bits; ++b)
        ta_state[b] &= ~(mask & carry);
}

static inline unsigned int cb_calculate_clause_output_update(
    const unsigned int *ta_state, 
    int number_of_ta_chunks, 
    int number_of_state_bits, 
    unsigned int filter, 
    const unsigned int *Xi)
{
    CB_VALIDATE_INPUTS_INT(ta_state && Xi && number_of_ta_chunks > 0 && number_of_state_bits > 0, 0);
    
    const int state_offset = number_of_state_bits - 1;
    unsigned int mismatch = 0;

    for (int k = 0; k < number_of_ta_chunks; ++k) {
        const unsigned int pos = k * number_of_state_bits + state_offset;
        const unsigned int ta_state_val = (k == number_of_ta_chunks - 1)
            ? ta_state[pos] & filter
            : ta_state[pos];
        mismatch |= (ta_state_val & Xi[k]) ^ ta_state_val;
    }

    return mismatch == 0;
}

int cb_number_of_include_actions(
        const unsigned int *ta_state,
        int clause,
        int number_of_literals,
        int number_of_state_bits
)
{
    CB_VALIDATE_INPUTS_INT(ta_state && clause >= 0 && number_of_literals > 0 && 
                          number_of_state_bits > 0, 0);
    
    const unsigned int filter = cb_calculate_filter(number_of_literals);
    const unsigned int number_of_ta_chunks = cb_calculate_ta_chunks(number_of_literals);
    const unsigned int clause_pos = clause * number_of_ta_chunks * number_of_state_bits;
    const int state_offset = number_of_state_bits - 1;

    int number_of_include_actions = 0;
    
    // Process all chunks except the last one
    for (int k = 0; k < (int)number_of_ta_chunks - 1; ++k) {
        const unsigned int ta_pos = k * number_of_state_bits + state_offset;
        number_of_include_actions += __builtin_popcount(ta_state[clause_pos + ta_pos]);
    }
    
    // Process the last chunk with filter
    const unsigned int last_ta_pos = (number_of_ta_chunks - 1) * number_of_state_bits + state_offset;
    number_of_include_actions += __builtin_popcount(ta_state[clause_pos + last_ta_pos] & filter);

    return number_of_include_actions;
}

void cb_clause_update(
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_state_bits,
    unsigned int *clause_output,
    const unsigned int *Xi,
    const int *Wi,
    unsigned int Y,
    int T,
    float s
)
{
    unsigned int max_included_literals = 3;
    const int use_sparse_feedback = (s > 1.0f);
    const unsigned int number_of_ta_chunks = cb_calculate_ta_chunks(number_of_literals);
    /* Allocate feedback_to_ta only if needed and initialize it once per update */
    unsigned int *feedback_to_ta = NULL;
    if (use_sparse_feedback) {
        feedback_to_ta = (unsigned int *)malloc(number_of_ta_chunks * sizeof(unsigned int));
        if (!feedback_to_ta) {
            /* Allocation failed: fall back to dense behavior */
            feedback_to_ta = NULL;
        } else {
            cb_initialize_random_streams(feedback_to_ta, number_of_literals, number_of_ta_chunks, s);
        }
    }

    CB_VALIDATE_INPUTS(ta_state && clause_output && Xi && Wi &&
                             number_of_clauses > 0 && number_of_literals > 0 && 
                             number_of_state_bits > 0);
    
    const unsigned int filter = cb_calculate_filter(number_of_literals);

    for (int j = 0; j < number_of_clauses; j++) {
        const unsigned int clause_pos = j * number_of_ta_chunks * number_of_state_bits;
        clause_output[j] = cb_calculate_clause_output_update(&ta_state[clause_pos], number_of_ta_chunks, number_of_state_bits, filter, Xi);

        int class_sum = Wi[j] * clause_output[j];
        float update_p = 0;
        if (Y == 1)
        {
            update_p = ((float)(T - class_sum)) / (2.0f * T);
            if (Wi[j] < 0 || !cb_should_update(update_p)) {
                continue;
            }
    
            const unsigned int clause_pos = j * number_of_ta_chunks * number_of_state_bits;
            if (clause_output[j] && cb_number_of_include_actions(ta_state, j, number_of_literals, number_of_state_bits) <= max_included_literals) {
                // Type Ia Feedback - Reinforce correct behavior
                for (int k = 0; k < number_of_ta_chunks; ++k) {
                    const unsigned int ta_pos = k * number_of_state_bits;
                    unsigned int *clause_ta_state = &ta_state[clause_pos + ta_pos];
                    const unsigned int xi_val = Xi[k];
                    
                    // Increment for included literals
                    cb_inc(clause_ta_state, xi_val, number_of_state_bits);
                    
                    // Decrement for excluded literals
                    const unsigned int excluded = ~xi_val;
                    if (use_sparse_feedback && feedback_to_ta) {
                        cb_dec(clause_ta_state, excluded & feedback_to_ta[k], number_of_state_bits);
                    } else {
                        cb_dec(clause_ta_state, excluded, number_of_state_bits);
                    }
                }
            } else {
                // Type Ib Feedback - Penalize incorrect behavior
                for (int k = 0; k < number_of_ta_chunks; ++k) {
                    const unsigned int ta_pos = k * number_of_state_bits;
                    unsigned int *clause_ta_state = &ta_state[clause_pos + ta_pos];
                    if (use_sparse_feedback && feedback_to_ta) {
                        cb_dec(clause_ta_state, feedback_to_ta[k], number_of_state_bits);
                    } else {
                        cb_dec(clause_ta_state, 0xffffffffU, number_of_state_bits);
                    }
                }
            }
        }
        else{
            update_p = ((float)(T + class_sum)) / (2.0f * T);
    
            // Skip inactive clauses or clauses not selected for update
            if (Wi[j] < 0 || !cb_should_update(update_p)) {
                continue;
            }

            const unsigned int clause_pos = j * number_of_ta_chunks * number_of_state_bits;
            if (clause_output[j]) {
                // Type II Feedback - Penalize false positives
                for (int k = 0; k < number_of_ta_chunks; ++k) {
                    const unsigned int ta_pos = k * number_of_state_bits;
                    // Increment automata for excluded literals (opposite of included)
                    cb_inc(&ta_state[clause_pos + ta_pos], ~Xi[k], number_of_state_bits);
                }
            }
        }
    
    }
    /* Free allocated feedback buffer if any */
    if (feedback_to_ta) {
        free(feedback_to_ta);
        feedback_to_ta = NULL;
    }
}

void tmae_train(
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
    float s
)
{
    // Allocate an example buffer once (will be reused for each produced example)
    unsigned int *X = malloc(sizeof(unsigned int) * number_of_ta_chunks);
    if (X == NULL) {
        // Allocation failed; nothing to do
        return;
    }

    int *clause_weight = (int *) malloc(sizeof(int) * (size_t)number_of_clauses);
    if (clause_weight == NULL) {
        free(X);
        return;
    }

    for (int ex = 0; ex < number_of_examples; ++ex){
        for (int target = 0; target < number_of_classes; ++target){

            int class_weight_start_index = target * number_of_clauses;
            for (int clause_index = 0; clause_index < number_of_clauses; ++clause_index) {
                clause_weight[clause_index] = class_weights[class_weight_start_index + clause_index];
            }

            int target_value = fast_rand() & 1; 

            produce_autoencoder_example(
                classes, 
                number_of_classes,
                indptr_row,
                indices_row,
                number_of_rows,
                indptr_col,
                indices_col,
                number_of_cols,
                X,
                target,
                target_value,
                accumulation
            );
            
            cb_clause_update(
                ta_state,
                number_of_clauses,
                number_of_literals,
                number_of_state_bits,
                clause_output,
                X,
                clause_weight,
                target_value,
                T,
                s
            );
        }
    }
    free(X);
    free(clause_weight);
}