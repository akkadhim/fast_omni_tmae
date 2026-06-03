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
);