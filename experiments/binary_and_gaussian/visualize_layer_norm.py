import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from strnn.models.adaptive_layer_norm import AdaptiveLayerNorm
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
import torch
from torch.nn.functional import softmax



def sample_data(h_dim, batch_size=10000):
    # Sample data points from Gaussian
    x = torch.tensor(
        np.random.normal(
            loc=0.0,
            scale=5.0,
            size=(batch_size, h_dim)
        ), dtype=torch.float32
    )
    return x


def plot_norm_first_v_weight_first():
    h_dim = 3
    batch_size = 10000
    x = sample_data(h_dim, batch_size)

    # Intiailize regular layer norm object
    layer_norm = torch.nn.LayerNorm(h_dim, elementwise_affine = False)

    # Intialize layer norm objects
    norm_first = AdaptiveLayerNorm(gamma=0.5, wp=1.0, normalize_first=True)
    weight_first = AdaptiveLayerNorm(gamma=0.5, wp=1.0, normalize_first=False)

    # Intialize structure and set norm weights
    mask_so_far = np.array([
        [3., 0., 0.],
        [0., 3., 0.],
        [3., 3., 3.]
    ])
    norm_first.set_norm_weights(mask_so_far)
    weight_first.set_norm_weights(mask_so_far)

    # Pass data through layer norm objects
    y_norm_first = norm_first(x)
    y_weight_first = weight_first(x)
    reg_layer_norm = layer_norm(x)

    # Visualize unonormalized and normalized data on a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='r', marker='o', label='Unnormalized')
    ax.scatter(
        y_norm_first[:, 0], 
        y_norm_first[:, 1], 
        y_norm_first[:, 2], 
        c='b', marker='o', alpha=0.5, label='norm_first'
    )
    ax.scatter(
        y_weight_first[:, 0], 
        y_weight_first[:, 1], 
        y_weight_first[:, 2], 
        c='g', marker='o', alpha=0.5, label='weight_first'
    )
    # ax.scatter(
    #     reg_layer_norm[:, 0], 
    #     reg_layer_norm[:, 1], 
    #     reg_layer_norm[:, 2], 
    #     c='r', marker='o', alpha=0.5, label='regular'
    # )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig("layer_norm_visualization2.png")


def plot_diff_gammas_2D():
    def set_labels(ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Same as plot_diff_gammas_3D but in 2D
    h_dim = 2
    batch_size = 10000
    x = sample_data(h_dim, batch_size)

    # Initialize structure
    mask_so_far = np.array([
        [3., 0.],
        [3., 3.]
    ])

    # Initialize layer norm object and set mask
    layer_norm = AdaptiveLayerNorm(wp=1.0)
    layer_norm.set_mask(mask_so_far)
    gammas = [0.1, 0.5, 0.8, 0.9, 1.0, 2.0, 3.0, 5.0, 10.0]
    num_cols = len(gammas) + 2

    # Initialize plot
    fig = plt.figure(figsize=(24, 3))

    for idx, gamma in enumerate(gammas):
        # Pass data through layer norm objects
        y = layer_norm(gamma, x)

        # Visualize normalized data on a 3D scatter plot   
        ax = fig.add_subplot(1, num_cols, idx + 1)
        ax.scatter(
            y[:, 0], y[:, 1],
            c='g', marker='o', alpha=0.5, label='adaptive'
        )
        ax.set_title(f'Gamma = {gamma}')
        set_labels(ax)

    # Add regular layer norm results to plot
    y = torch.nn.LayerNorm(h_dim, elementwise_affine = False)(x)
    ax = fig.add_subplot(1, num_cols, num_cols - 1)
    ax.scatter(
        y[:, 0], y[:, 1],
        c='g', marker='o', alpha=0.5, label='regular'
    )
    ax.set_title('Regular Layer Norm')
    set_labels(ax)

    # Add original points (x) to plot
    ax = fig.add_subplot(1, num_cols, num_cols)
    ax.scatter(
        x[:, 0], x[:, 1],
        c='g', marker='o', alpha=0.2, label='original'
    )
    ax.set_title('Original Data')
    set_labels(ax)

    plt.legend()
    plt.savefig(f"layer_norm_vis_diff_gammas_2D.png")



def plot_diff_gammas_3D():
    def set_labels(ax):
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    h_dim = 3
    batch_size = 10000
    x = sample_data(h_dim, batch_size)

    # Initialize structure
    mask_so_far = np.array([
            [3., 0., 0.],
            [0., 3., 0.],
            [3., 3., 3.]
        ])
    
    # Initialize layer norm object and set mask
    adapt_layer_norm = AdaptiveLayerNorm(wp=1.0)
    adapt_layer_norm.set_mask(mask_so_far)
    gammas = [0.1, 0.5, 0.8, 0.9, 1.0, 2.0, 3.0, 5.0, 10.0]
    num_cols = len(gammas) + 2

    # Initialize plot
    fig = plt.figure(figsize=(25, 6))

    for idx, gamma in enumerate(gammas):
        # Pass data through layer norm objects
        y = adapt_layer_norm(gamma, x)

        # Visualize normalized data on a 3D scatter plot   
        ax = fig.add_subplot(1, num_cols, idx + 1, projection='3d')
        ax.scatter(
            y[:, 0], y[:, 1], y[:, 2],
            c='g', marker='o', alpha=0.5, label='adaptive'
        )
        ax.set_title(f'Gamma = {gamma}')
        set_labels(ax)

    # Add regular layer norm results to plot
    layer_norm = torch.nn.LayerNorm(h_dim, elementwise_affine = False)
    y = layer_norm(x)
    ax = fig.add_subplot(1, num_cols, num_cols - 1, projection='3d')
    ax.scatter(
        y[:, 0], y[:, 1], y[:, 2],
        c='g', marker='o', alpha=0.5, label='regular'
    )
    ax.set_title('Regular Layer Norm')
    set_labels(ax)

    # Add original points (x) to plot
    ax = fig.add_subplot(1, num_cols, num_cols, projection='3d')
    ax.scatter(
        x[:, 0], x[:, 1], x[:, 2], 
        c='g', marker='o', alpha=0.2, label='original'
    )
    ax.set_title('Original Data')
    set_labels(ax)

    plt.legend()
    plt.savefig(f"layer_norm_vis_diff_gammas_3D.png")


def plot_normalized_gaussian():
    # Generate 10000 random points from a 2D Gaussian distribution
    np.random.seed(0)  # For reproducibility
    points = np.random.randn(10000, 2)

    # Compute the sample mean and variance
    mean = points.mean(axis=1, keepdims=True)
    var = points.var(axis=1, keepdims=True)

    # Normalize the points
    normalized_points = (points - mean) / np.sqrt(var + 1e-5)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original points
    ax1.scatter(points[:, 0], points[:, 1], alpha=0.5)
    ax1.set_title("Original")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis('equal')

    # Plot normalized points
    ax2.scatter(normalized_points[:, 0], normalized_points[:, 1], alpha=0.5, color='red')
    ax2.set_title("Normalized")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig("2d_layer_norm.png")


def gradient_flow_investigation():
    model = StrNNDensityEstimator()


def try_gamma(gamma=1.0):
    connections = torch.Tensor([3, 6, 6, 3])
    norm_weights = softmax(connections / gamma, dim=0)
    return norm_weights


if __name__ == '__main__':
    # plot_normalized_gaussian()
    plot_diff_gammas_2D()
    # plot_diff_gammas_3D()
    # for g in [0.1, 0.5, 0.8, 0.9, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
    #     print(f"{g}: {try_gamma(g)}")




