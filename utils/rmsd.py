import jax.numpy as jnp
import jax

"""
From https://hunterheidenreich.com/posts/kabsch_algorithm/ and adapted
"""


@jax.jit
def kabsch_align(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: Return aligned P and Q
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = jnp.dot(p.T, q)

    # SVD
    U, S, Vt = jnp.linalg.svd(H)

    # Validate right-handed coordinate system
    Vt = jnp.where(jnp.linalg.det(jnp.dot(Vt.T, U.T)) < 0.0, -Vt, Vt)

    # Optimal rotation
    R = jnp.dot(Vt.T, U.T)

    return jnp.dot(p, R.T), q


@jax.jit
def kabsch_rmsd(P, Q):
    P_aligned, Q_aligned = kabsch_align(P, Q)
    return jnp.sqrt(jnp.sum(jnp.square(P_aligned - Q_aligned)) / P.shape[0])
