import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from fast_tmae.clause_bank import ClauseBank

class TMAutoEncoder:
    def __init__(
            self,
            number_of_clauses,
            T,
            s,
            output_active,
            accumulation=1,
            max_included_literals=None,
            number_of_state_bits_ta=8,
            seed=None,
            batch_size=100,
            platform="CPU",
            backend="cpu",
            device_ids=None,
            **kwargs  # Catch any remaining unused parameters
    ):
        self.output_active = output_active
        self.accumulation = accumulation
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.number_of_clauses = number_of_clauses
        self.number_of_state_bits_ta = number_of_state_bits_ta
        self.T = int(T)
        self.s = s
        self.platform = platform
        self.backend = backend  # 'cpu', 'cuda', or 'opencl'
        self.device_ids = device_ids
        
        self.max_included_literals = max_included_literals
        self.batch_size = batch_size
        
        self.X_train = np.zeros(0, dtype=np.uint32)
        self.X_test = np.zeros(0, dtype=np.uint32)
        self._X_train_input = None
        self.initialized = False
        self.clause_bank = None
        
    def init(self, X: np.ndarray, Y: np.ndarray = None):
        if self.initialized:
            return
        
        self.number_of_classes = self.output_active.shape[0]
        self.clause_bank = ClauseBank(
            seed=self.seed,
            X_shape=X.shape,
            s=self.s,
            T=self.T,
            max_included_literals=self.max_included_literals,
            number_of_clauses=self.number_of_clauses,
            number_of_state_bits_ta=self.number_of_state_bits_ta,
            batch_size=self.batch_size,
            output_active=self.output_active,
            platform=self.platform,
            backend=self.backend,
            device_ids=self.device_ids
        )
        if self.max_included_literals is None:
            self.max_included_literals = self.clause_bank.number_of_literals
          
        output_balancing = 0.5
        self.feature_true_probability = np.ones(X.shape[1], dtype=np.float32) * output_balancing
        self.initialized = True

    def fit(self, X, number_of_epochs=1, number_of_examples=2000, shuffle=True, *args, **kwargs):
        using_shared_sparse_input = isinstance(X, tuple) and len(X) == 2

        if using_shared_sparse_input:
            X_csr, X_csc = X
        else:
            X_csr = csr_matrix(X.reshape(X.shape[0], -1))
            X_csc = csc_matrix(X.reshape(X.shape[0], -1)).sorted_indices()

        self.init(X_csr, Y=None)

        if using_shared_sparse_input:
            if self._X_train_input is not X:
                self.encoded_X_train = self.clause_bank.prepare_X_autoencoder(X_csr, X_csc, self.output_active)
                self._X_train_input = X
        else:
            X_train_signature = np.concatenate((X_csr.indptr, X_csr.indices))
            if not np.array_equal(self.X_train, X_train_signature):
                self.encoded_X_train = self.clause_bank.prepare_X_autoencoder(X_csr, X_csc, self.output_active)
                self.X_train = X_train_signature
                self._X_train_input = X

        if self.platform == "CPU":
            self.clause_bank.train_cpu(
                        number_of_examples=number_of_examples,
                        encoded_X=self.encoded_X_train,
                        accumulation=self.accumulation,
                    )
        else:
            self.clause_bank.train_gpu(
                        number_of_epochs=number_of_epochs,
                        number_of_examples=number_of_examples,
                        encoded_X=self.encoded_X_train,
                        accumulation=self.accumulation,
                    )

    def finish(self):
        pass
    
    def get_embeddings(self):
        return self.clause_bank.embeddings

    def get_weights(self, the_class):
        """Get weights for a specific class"""
        return self.clause_bank.get_weights(the_class)
    
    def get_ta_state(self, clause, ta):
        """Get TA state for a specific clause and literal"""
        return self.clause_bank.get_ta_state(clause, ta)
