"""
Utility functions for a simple Bernoulli RBM using NumPy and scikit-learn's
built-in digits dataset. Everything is kept simple and commented so it is easy
to follow.
"""

import os
from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom


@dataclass
class RBMParams:
    """Container for RBM parameters."""
    W: np.ndarray  # (nv, nh) weight matrix
    b: np.ndarray  # (nv,) visible bias
    c: np.ndarray  # (nh,) hidden bias


def load_digits_data(target_size=(14, 14), threshold=0.5, test_size=0.2, seed=0):
    """
    Load the built-in 8x8 digits dataset, optionally resize to target_size,
    scale to [0,1], binarize, and split into train/test sets.
    """
    digits = load_digits()
    images = digits.images  # shape: (n_samples, 8, 8), values 0-16

    # scale original pixel values (0-16) to [0,1]
    images = images / 16.0

    # resize to target_size if needed (uses simple scipy zoom)
    if target_size != (8, 8):
        row_factor = target_size[0] / images.shape[1]
        col_factor = target_size[1] / images.shape[2]
        images = zoom(images, (1, row_factor, col_factor), order=1)

    # flatten and binarize
    flat = images.reshape(len(images), -1)
    binary = (flat > threshold).astype(np.float32)

    X_train, X_test = train_test_split(
        binary, test_size=test_size, random_state=seed, shuffle=True
    )
    return X_train, X_test, target_size


def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def sample_bernoulli(probabilities):
    """Sample 0/1 given probabilities."""
    return (np.random.rand(*probabilities.shape) < probabilities).astype(np.float32)


def hamming_error(true_bits, pred_bits):
    """Average fraction of mismatched bits."""
    mismatches = np.not_equal(true_bits, pred_bits)
    return mismatches.mean()


def rbm_cd1_train(
    train_data,
    test_data,
    nv,
    nh=50,
    epochs=50,
    batch_size=100,
    lr=0.1,
    weight_decay=1e-4,
    seed=0,
):
    """
    Train a Bernoulli RBM with CD-1.
    Returns: params (W,b,c), train_errors list, test_errors list.
    """
    np.random.seed(seed)
    num_train = train_data.shape[0]

    # Initialize weights/biases
    W = 0.01 * np.random.randn(nv, nh).astype(np.float32)
    b = np.zeros(nv, dtype=np.float32)
    c = np.zeros(nh, dtype=np.float32)

    train_errors = []
    test_errors = []

    for epoch in range(1, epochs + 1):
        # Shuffle data each epoch
        perm = np.random.permutation(num_train)
        train_shuffled = train_data[perm]

        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            v0 = train_shuffled[start:end]
            m = v0.shape[0]

            # Positive phase
            p_h0 = sigmoid(np.dot(v0, W) + c)
            h0 = sample_bernoulli(p_h0)

            # Negative phase (reconstruct)
            p_v1 = sigmoid(np.dot(h0, W.T) + b)
            v1 = sample_bernoulli(p_v1)
            p_h1 = sigmoid(np.dot(v1, W) + c)

            # Gradients
            pos_grad = np.dot(v0.T, p_h0)
            neg_grad = np.dot(v1.T, p_h1)

            # Parameter updates
            W += lr * ((pos_grad - neg_grad) / m - weight_decay * W)
            b += lr * np.mean(v0 - v1, axis=0)
            c += lr * np.mean(p_h0 - p_h1, axis=0)

        # Track errors on subsets
        train_recon = rbm_reconstruct(train_data[: min(500, num_train)], W, b, c)
        train_err = hamming_error(train_data[: train_recon.shape[0]], train_recon)
        train_errors.append(train_err)

        if test_data is not None and len(test_data) > 0:
            test_recon = rbm_reconstruct(test_data[: min(500, len(test_data))], W, b, c)
            test_err = hamming_error(test_data[: test_recon.shape[0]], test_recon)
        else:
            test_err = np.nan
        test_errors.append(test_err)

        print(
            f"Epoch {epoch}/{epochs} - train err {train_err:.4f}"
            f" - test err {test_err:.4f}"
        )

    params = RBMParams(W=W, b=b, c=c)
    return params, train_errors, test_errors


def rbm_reconstruct(v, W, b, c):
    """One-step reconstruction v -> h -> v'."""
    p_h = sigmoid(np.dot(v, W) + c)
    h = sample_bernoulli(p_h)
    p_v = sigmoid(np.dot(h, W.T) + b)
    return (p_v > 0.5).astype(np.float32)


def rbm_gibbs_step(v, W, b, c):
    """One full Gibbs step."""
    p_h = sigmoid(np.dot(v, W) + c)
    h = sample_bernoulli(p_h)
    p_v = sigmoid(np.dot(h, W.T) + b)
    return (p_v > 0.5).astype(np.float32)


def corrupt_bits(v, noise_level):
    """Flip each bit with probability noise_level."""
    flip_mask = np.random.rand(*v.shape) < noise_level
    return np.logical_xor(v.astype(bool), flip_mask).astype(np.float32)


def save_model(params, target_size, path):
    """Save RBM parameters and target_size to an .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, W=params.W, b=params.b, c=params.c, target_size=target_size)


def load_model(path):
    """Load RBM parameters and target_size from an .npz file."""
    data = np.load(path)
    params = RBMParams(W=data["W"], b=data["b"], c=data["c"])
    target_size = tuple(data["target_size"])
    return params, target_size
