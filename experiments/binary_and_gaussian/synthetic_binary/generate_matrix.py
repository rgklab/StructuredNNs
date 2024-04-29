import numpy as np
import pandas as pd
import torch
import os
from numpy.random import binomial, standard_normal
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
import tensorflow_datasets as tfds

# Data should be numpy arrays of shape [number of data points, data dimensions]

NUM_TEST_SAMPLES = 500
RANDOM_THRESHOLD = {'middense': 0.2, 'dense': 0.4, 'medium': 0.6, 'mediumsparse': 0.75, 'sparse': 0.9, 'verysparse': 0.95}


class DataGenerator():

    def __init__(self):
        pass

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
        @return: d by d adjacency matrix
        """
        if adj_type == 'full_auto':
            return np.tril(np.tril(np.ones((d, d)), -1))
        elif adj_type == 'full_ones':
            return np.ones((d, d))
        elif 'prev' in adj_type:
            # Parse adj_type for how many previous dimensions to consider
            K = int(adj_type.split('_')[1])

            if K < d - 1:
                # Each node depends on the K nodes that came before
                A = np.zeros((d, d))
                for i in range(1, d):
                    for k in range(1, K + 1):
                        col = i - k
                        if col >= 0:
                            A[i][col] = 1
                return A
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
                print(f"Adj mtx generation attempt {attempt_num}:")
                A = np.random.rand(d, d)  # Use rand for [0, 1) uniform distribution
                A = np.tril(A, -1)  # Make lower triangular
                A[A > threshold] = 1  # Set connections
                A[A <= threshold] = 0  # Remove connections to achieve sparsity

                # Check each node has at least one dependency (except the first node)
                if np.all(A.sum(axis=1)[1:] > 0):
                    print("Attempt successful - very sparse adj mtx generated!")
                    return A
                else:
                    print("Attempt failed; retry ...")
                    attempt_num += 1

                print("Attempt successful - random adj mtx generated!")
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
                        p_i = 1 / (1 + np.exp(-p_i))
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

    def save_data(self, data_type, adj_type, data_dim, num_samples):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_dir, 'synth_data_files')
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, f"{data_type}_{adj_type}_d{str(data_dim)}_n{str(num_samples)}.npz")
        print(f"Saving data to: {file_path}")
        np.savez(
            file_path,
            train_data=self.train_data,
            valid_data=self.val_data,
            test_data=self.test_data
        )

    def generate_data_group(self, data_type, adj_type, data_dim, sample_sizes):
        # Create and save adjacency matrix
        A = self.create_adjacency(
            d=data_dim, adj_type=adj_type
        )
        current_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.join(current_dir, 'synth_data_files')
        os.makedirs(target_dir, exist_ok=True)
        file_path = os.path.join(target_dir, f"{adj_type}_d{str(data_dim)}_adj.npz")
        np.savez(
            file_path,
            A
        )

        # Make sure sample_sizes are sorted in descending order
        # sample_sizes.sort(reverse=True)
        max_sample_size = sample_sizes[0]

        # Generate train_val and test data
        print("Generating data ...")
        self.generate_data(
            data_type, adj_type, A, data_dim=data_dim, num_samples=max_sample_size
        )

        print("Processing by sample_size ...")
        # Process by sample_size
        for sample_size in sample_sizes:
            if sample_size == max_sample_size:
                assert len(self.train_val_data) == max_sample_size
                sub_train_val_data = self.train_val_data
            else:
                sub_train_val_data = self.train_val_data[: sample_size]
            self.train_data, self.val_data = train_test_split(
                sub_train_val_data, test_size=0.2, random_state=42
            )
            print(f"Saving data sample_size {sample_size}")
            print("train size: " + str(len(self.train_data)))
            print("val size: " + str(len(self.val_data)))
            print("test size: " + str(len(self.test_data)))
            self.save_data(data_type, adj_type, data_dim, sample_size)


def download_minst():
    mnist_data = tfds.load("binarized_mnist")
    mnist_train, mnist_validation, mnist_test = mnist_data["train"], mnist_data['validation'], mnist_data["test"]

    def process_dataset(dataset):
        dataset = list(tfds.as_numpy(dataset))

        def func(x):
            img = x['image']
            img = img.flatten()
            return img

        dataset = np.asarray(list(map(func, dataset)))
        return dataset

    train_data = process_dataset(mnist_train)
    validation_data = process_dataset(mnist_validation)
    test_data = process_dataset(mnist_test)

    np.savez(
        "binary_MNIST",
        train_data=train_data,
        valid_data=validation_data,
        test_data=test_data
    )


def subsample_MNIST(samples):
    assert samples <= 50000, "Not enough training samples to start with!"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, *['..', 'MADE', 'datasets'])
    data = np.load(os.path.join(dir_path, 'binary_MNIST.npz'))

    train_data = data['train_data'].astype(np.float32)
    valid_data = data['valid_data'].astype(np.float32)
    test_data = data['test_data'].astype(np.float32)

    # Shuffle and subsample training data
    np.random.seed(42)
    np.random.shuffle(train_data)
    train_data = train_data[:samples]

    # Save data
    filename = f"binary_MNIST_{samples}"
    np.savez(
        filename,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data
    )
    print("Saved " + filename)


if __name__ == '__main__':
    # Initialize the DataGenerator
    gen = DataGenerator()
    data_dim = 28  # or any other appropriate dimension based on your needs
    fully_connected_adj_mtx = gen.create_adjacency(d=data_dim, adj_type='full_ones')
    np.savez(f"./synth_data_files/full_ones_adj_mtx_{data_dim}.npz", A=fully_connected_adj_mtx)

    # Generate data with the fully connected adjacency matrix if needed
    sample_sizes = [2000, 1000]  # Example sample sizes
    gen.generate_data_group(data_type='binary', adj_type='full_ones', data_dim=data_dim, sample_sizes=sample_sizes)
