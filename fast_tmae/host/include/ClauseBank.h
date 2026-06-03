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
);

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
    int *classes_weights,
    unsigned int *ta_state,
    int number_of_clauses,
    int number_of_literals,
    int number_of_ta_chunks, 
    int number_of_state_bits,
    unsigned int *clause_output,
    int T,
    float s
);