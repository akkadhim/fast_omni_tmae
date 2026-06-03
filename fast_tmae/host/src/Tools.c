#include <string.h>
#include <stdlib.h>
#include "fast_rand.h"

// Inline helper functions for bit manipulation
static inline void set_feature_bit(unsigned int *X, int feature_idx) {
    const int chunk_nr = feature_idx / 32;
    const int chunk_pos = feature_idx % 32;
    X[chunk_nr] |= (1U << chunk_pos);
}

static inline void clear_feature_bit(unsigned int *X, int feature_idx) {
    const int chunk_nr = feature_idx / 32;
    const int chunk_pos = feature_idx % 32;
    X[chunk_nr] &= ~(1U << chunk_pos);
}

static inline void process_feature_pair(unsigned int *X, int feature_idx, int number_of_features) {
    // Set the positive literal bit
    set_feature_bit(X, feature_idx);
    // Clear the negative literal bit
    clear_feature_bit(X, feature_idx + number_of_features);
}

int compareints(const void *a, const void *b) {
    const unsigned int *ia = (const unsigned int *)a;
    const unsigned int *ib = (const unsigned int *)b;
    if (*ia < *ib) return -1;
    if (*ia > *ib) return 1;
    return 0;
}


void produce_autoencoder_example(
        const unsigned int *active_output,
        int number_of_active_outputs,
        const unsigned int *indptr_row,
        const unsigned int *indices_row,
        int number_of_rows,
        unsigned int *indptr_col,
        unsigned int *indices_col,
        int number_of_cols,
        unsigned int *X,
        int target,
        int target_value,
        int accumulation
)
{
    // Input validation
    if (!active_output || !indptr_row || !indices_row || !indptr_col || 
        !indices_col || !X || number_of_cols <= 0 || number_of_rows <= 0 || 
        target < 0 || accumulation <= 0) {
        return;
    }

    const int number_of_features = number_of_cols;
    const int number_of_literals = 2 * number_of_features;
    const unsigned int number_of_literal_chunks = (number_of_literals - 1) / 32 + 1;
    
    // Cache frequently accessed values
    const unsigned int target_start = indptr_col[active_output[target]];
    const unsigned int target_end = indptr_col[active_output[target] + 1];
    const unsigned int target_size = target_end - target_start;
    
    int row;

    // Initialize example vector X - set all negative literal bits to 1
    memset(X, 0, number_of_literal_chunks * sizeof(unsigned int));
    
    // Optimized initialization: set negative literal bits (second half)
    const int neg_start_chunk = number_of_features / 32;
    const int neg_start_pos = number_of_features % 32;
    
    // Fill complete chunks with all 1s
    for (int chunk = neg_start_chunk + (neg_start_pos ? 1 : 0); 
         chunk < (int)number_of_literal_chunks && chunk * 32 < number_of_literals; ++chunk) {
        const int bits_in_chunk = (chunk * 32 + 32 <= number_of_literals) ? 32 : 
                                 (number_of_literals - chunk * 32);
        X[chunk] = (bits_in_chunk == 32) ? 0xFFFFFFFFU : ((1U << bits_in_chunk) - 1);
    }
    
    // Handle partial first chunk if needed
    if (neg_start_pos > 0) {
        const unsigned int mask = 0xFFFFFFFFU << neg_start_pos;
        const int bits_in_chunk = (neg_start_chunk * 32 + 32 <= number_of_literals) ? 
                                 32 : (number_of_literals - neg_start_chunk * 32);
        const unsigned int end_mask = (bits_in_chunk < 32) ? ((1U << bits_in_chunk) - 1) : 0xFFFFFFFFU;
        X[neg_start_chunk] |= (mask & end_mask);
    }
	
    // Check for edge cases: no positive examples or all examples are positive
    if (target_size == 0 || target_size == (unsigned int)number_of_rows) {
        // If no positive/negative examples, produce random examples
        for (int a = 0; a < accumulation; ++a) {
            row = (int)(fast_rand() % (unsigned int)number_of_rows);
            for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
                process_feature_pair(X, indices_row[k], number_of_features);
            }
        }
        return;
    }

    if (target_value) {
        // Generate examples from positive samples
        for (int a = 0; a < accumulation; ++a) {
            // Pick example randomly among positive examples
            const int random_offset = (int)(fast_rand() % target_size);
            const int random_index = target_start + random_offset;
            row = indices_col[random_index];
            
            // Process all features for this row
            for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
                process_feature_pair(X, indices_row[k], number_of_features);
            }
        }
    } else {
        // Generate examples from negative samples (not in positive set)
        int a = 0;
        while (a < accumulation) {
            row = (int)(fast_rand() % (unsigned int)number_of_rows);

            // Check if this row is NOT in the positive examples list
            if (bsearch(&row, &indices_col[target_start], target_size, 
                       sizeof(unsigned int), compareints) == NULL) {
                // Process all features for this negative example row
                for (int k = indptr_row[row]; k < indptr_row[row + 1]; ++k) {
                    process_feature_pair(X, indices_row[k], number_of_features);
                }
                a++;
            }
        }
    }
}
