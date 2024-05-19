"""
This file contains data generators for the synthetic binary and Gaussian 
datasets used in StrNN experiments. Datasets are stored as .npz files, 
split into train, validation, and test sets.
"""

import numpy as np
from numpy.random import binomial, standard_normal
from numpy.linalg import inv
from sklearn.model_selection import train_test_split


RANDOM_THRESHOLD = {
    'dense': 0.5, 
    'medium-dense': 0.6, 
    'medium': 0.7, 
    'medium-sparse': 0.8,
    'sparse': 0.9}
NUM_TEST_SAMPLES = 500


class DataGenerator():

    def __init__(self, data_path):
        self.data_path = data_path

    def create_adjacency(self, d, adj_type='full_auto'):
        """
        Construct adjacency matrix A
        A is strictly lower triangular with all 1s below the diagonal
        for fully autoregressive data

        @param d: int, data dimension
        @param adj_type:
            full_auto: default, fully autoregressive, A is all ones below diagonal
            prev_k: x_i only depends on the k dimensions that came before
                 when k=1, sequence has Markov property
            every_other: skips every other connection
            random_y: randomly zeroes out y% of the connections
            only_first_k: each x_i depends on only the first k dimensions 
        @return: d by d adjacency matrix
        """
        if adj_type == 'full_auto':
            return np.tril(np.tril(np.ones((d, d)), -1))
        elif 'prev' in adj_type:
            # Parse adj_type for how many previous dimensions to consider
            K = int(adj_type.split('_')[1])

            if K < d - 1:
                # Each node depends on the K nodes that came before
                A = np.zeros((d, d))
                for i in range(1, d):
                    for k in range(1, K+1):
                        col = i - k
                        if col >= 0:
                            A[i][col] = 1
                
            elif K == d - 1:
                # This case is equivalent to fully connected lower triangular A
                return self.create_adjacency(d, adj_type='full_auto')
            else:
                raise ValueError("Invalid prev_k argument!")
        elif adj_type == 'every_other':
            A = np.tril(np.tril(np.ones((d, d)), -1))
            for i in range(d):
                for j in range(d):
                    if i > j:
                        # Lower triangular; everything else is zero already
                        if (i - j) % 2 == 0:
                            A[i][j] = 0
            return A
        elif 'random' in adj_type:
            _, random_type = adj_type.split('_')
            assert random_type in RANDOM_THRESHOLD
            threshold = RANDOM_THRESHOLD[random_type]

            attempt_num = 0
            while True:
                # Parse adj_type for how many entries to randomly zero out

                # Initialize randomly
                print(f"Adj mtx generation attempt {attempt_num}:")
                A = np.random.standard_normal((d, d))
                # Threshold and make lower triangular
                for i in range(len(A)):
                    for j in range(len(A[0])):
                        if A[i][j] > -1*threshold and A[i][j] < threshold:
                            A[i][j] = 0
                        else:
                            A[i][j] = 1
                A = np.tril(A, -1)

                # Check each node has dependencies
                sums = A.sum(axis=1)
                has_empty_row = False
                for sum in sums[1:]:
                    if sum == 0:
                        has_empty_row = True
                if has_empty_row:
                    print("Attempt failed; retry ...")
                    attempt_num += 1
                    continue

                print("Attempt successful - random adj mtx generated!")
                return A
        elif 'only_first' in adj_type:
            k = int(adj_type.split('_')[-1])
            A = np.zeros((d, d))
            # Set first column (except first row) to one
            A[1:, 0] = 1
            return A
        else:
            raise ValueError(f"{adj_type} is not a valid adj_type!")
        

    def generate_data(self, type, adj_type, A, data_dim=10, num_samples=100, val_pct=0.2, test_pct=0.2):
        """
        Generates synthetic autoregressive data
        By default, x_i depends on x_1 to x_i-1
        Binary case:
        x_i ~ Bernoulli(p_i) where
        p_i = Sigmoid(f(x_1, ..., x_i-1))
        Gaussian case:
        x_i ~ N()

        @param type: 'binary' or 'real'
        @param d: int, dimension of data vector
        @param n: int, number of samples to generate
        @return: train_data, test_data
        """
        # Construct alpha matrix
        # Each entry is standard normal
        alpha = standard_normal((data_dim, data_dim))
        # Element-wise combine alpha with adjacency matrix
        alpha = np.multiply(alpha, A)

        def _gen_data(N):
            data = []

            if type == 'binary':
                for n in range(N):
                    x_n = np.zeros(data_dim)
                    x_n[0] = binomial(1, p=0.5)

                    for i in range(1, data_dim):
                        p_i = np.dot(alpha[i], x_n)
                        p_i = 1/(1 + np.exp(-p_i))
                        x_n[i] = binomial(1, p=p_i)

                    data.append(x_n)
            elif type == 'gaussian':
                # Sample constants
                c = standard_normal((data_dim, 1))
                sigma = standard_normal()

                # Sample noise vector and generate Gaussian data
                for n in range(N):
                    z = standard_normal((data_dim, 1))
                    # x = (I - Alpha)^-1 * (c + sigma * z)
                    x = np.matmul(inv(np.identity(data_dim) - alpha), c + sigma * z)
                    x = x.reshape((x.shape[0],))
                    data.append(x)
            else:
                raise ValueError(f"{type} is not a valid type (real or binary)")

            return data

        total_num_samples = num_samples + NUM_TEST_SAMPLES
        self.all_data = _gen_data(total_num_samples)

        # Split data into train and validation sets
        test_pct = NUM_TEST_SAMPLES / total_num_samples
        self.train_val_data, self.test_data = train_test_split(
            self.all_data, test_size=test_pct, random_state=42
        )
        self.train_data, self.val_data = train_test_split(
            self.train_val_data, test_size=0.2, random_state=42
        )


    def save_data(self, type, adj_type, data_dim, num_samples):
        """
        Save data and adj_mtx as npz files
        """
        # Save data splits
        np.savez(
            self.data_path + f"{type}_{adj_type}_d{str(data_dim)}_n{str(num_samples)}",
            train_data=self.train_data,
            valid_data=self.val_data,
            test_data=self.test_data
        )


def subsample_data(
        folder_path: str,
        dataset_name: str, 
        original_size: int,
        subsample_sizes: list[int]
    ):
    """
    Subsample training data and val data proportionally. Test size remains the same.
    """
    # Load data
    load_data_path = f"{dataset_name}_n{original_size}.npz"
    data = np.load(folder_path + load_data_path)
    train_data = data['train_data']
    val_data = data['valid_data']
    test_data = data['test_data']
    orig_size = train_data.shape[0] + val_data.shape[0] + test_data.shape[0]

    # Subsample data
    for size in subsample_sizes:
        ratio = size / orig_size
        train_size = int(train_data.shape[0] * ratio)
        val_size = int(val_data.shape[0] * ratio)
        # Randomly sample indices up to train_size and val_size
        train_indices = np.random.choice(train_data.shape[0], train_size, replace=False)
        val_indices = np.random.choice(val_data.shape[0], val_size, replace=False)
        # Subsample data
        train_data_sub = train_data[train_indices]
        val_data_sub = val_data[val_indices]

        # Save data
        save_data_name = f"{dataset_name}_n{size}.npz"
        np.savez(
            folder_path + save_data_name,
            train_data=train_data_sub,
            valid_data=val_data_sub,
            test_data=test_data
        )
        print(f"Subsampled data saved successfully at {save_data_name}.")


if __name__ == '__main__':
    data_path = '../experiments/binary_and_gaussian/synth_data_files/'
    data_gen = DataGenerator(data_path)
    data_dim = 500
    num_samples = 5000
    adj_type = 'only_first_1'
    adj_mtx = data_gen.create_adjacency(data_dim, adj_type)
    data_gen.generate_data('binary', adj_type, adj_mtx, data_dim, num_samples)
    data_gen.save_data('binary', adj_type, data_dim, num_samples)
    # Save adj mtx
    np.savez(data_path + f"{adj_type}_d{str(data_dim)}_adj", adj_mtx)
    print(f"Data saved successfully for {adj_type}.")

    # for sparsity_level in ['dense', 'medium-dense', 'medium', 'medium-sparse', 'sparse']:
    #     adj_type = f'random_{sparsity_level}'
    #     adj_mtx = data_gen.create_adjacency(data_dim, adj_type)
    #     data_gen.generate_data('binary', adj_type, adj_mtx, data_dim, num_samples)
    #     data_gen.save_data('binary', adj_type, data_dim, num_samples)
    #     # Save adj mtx
    #     np.savez(data_path + f"{adj_type}_d{str(data_dim)}_adj", adj_mtx)
    #     print(f"Data saved successfully for {adj_type}.")

    # folder_path = '../experiments/binary_and_gaussian/synth_data_files/'
    # dataset_name = 'binary_random_sparse_d1000'
    # original_size = 6000
    # subsample_sizes = [5000, 4000, 3000, 2000, 1000]
    # subsample_data(folder_path, dataset_name, original_size, subsample_sizes)