import jax.numpy as jnp
import jax

"""
From https://hunterheidenreich.com/posts/kabsch_algorithm/ and adapted
"""


@jax.jit
def kabsch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.

    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"

    # Compute centroids
    centroid_P = jnp.mean(P, axis=0)
    centroid_Q = jnp.mean(Q, axis=0)

    # Optimal translation
    t = centroid_Q - centroid_P

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

    return jnp.sqrt(jnp.sum(jnp.square(jnp.dot(p, R.T) - q)) / P.shape[0])
