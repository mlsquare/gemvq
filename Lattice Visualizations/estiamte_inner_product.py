import numpy as np
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_d4, calculate_mse
from closest_point import closest_point_Dn, custom_round


def pad_vector(vec, lattice_dim):
    """Pad the vector with zeros to make its length a multiple of lattice_dim."""
    remainder = len(vec) % lattice_dim
    if remainder != 0:
        padding = lattice_dim - remainder
        vec = np.concatenate([vec, np.zeros(padding)])
    return vec


def precompute_hq_lut(G, Q_nn, q):
    hq = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=1)
    codebook = hq.create_q_codebook()
    lookup_table = {}
    for enc1, lattice_point1 in codebook.items():
        for enc2, lattice_point2 in codebook.items():
            inner_product = np.dot(lattice_point1, lattice_point2)
            lookup_table[(enc1, enc2)] = inner_product
    return lookup_table


def estimate_inner_product(a, b, quantizer, method="nested", lookup_table=None):
    """Estimate the inner product of two vectors using the specified quantization method."""
    lattice_dim = quantizer.G.shape[0]
    a_padded = pad_vector(a, lattice_dim)
    b_padded = pad_vector(b, lattice_dim)

    inner_product = 0
    for i in range(0, len(a_padded), lattice_dim):
        a_block = a_padded[i:i + lattice_dim]
        b_block = b_padded[i:i + lattice_dim]

        if method == "nested":
            a_enc, a_i0 = quantizer.encode(a_block)
            a_decoded = quantizer.decode(a_enc, a_i0)
            b_enc, b_i0 = quantizer.encode(b_block)
            b_decoded = quantizer.decode(b_enc, b_i0)
            inner_product += np.dot(a_decoded, b_decoded)
        elif method == "hierarchical":
            a_enc = quantizer.encode(a_block)
            b_enc = quantizer.encode(b_block)

            a_enc_vectors, a_i0 = a_enc
            b_enc_vectors, b_i0 = b_enc

            a_l, a_m = (tuple(b.astype(int).tolist()) for b in a_enc_vectors)
            b_l, b_m = (tuple(b.astype(int).tolist()) for b in b_enc_vectors)

            inner_product += lookup_table[(a_l, b_l)]
            inner_product += quantizer.q * lookup_table[(a_m, b_l)]
            inner_product += quantizer.q * lookup_table[(a_l, b_m)]
            inner_product += quantizer.q ** 2 * lookup_table[(a_m, b_m)]
        else:
            raise ValueError("Unsupported method. Use 'nested' or 'hierarchical'.")

    return inner_product

def main():
    q = 5
    vector_dim = 128
    G = np.eye(2)
    q_nn = custom_round
    var = 3

    nested_quantizer = NQuantizer(G, q_nn, beta=1, q=q)
    hierarchical_quantizer = HQuantizer(G, q_nn, beta=1, q=q)

    lookup_table = precompute_hq_lut(G, q_nn, q)

    # a = np.random.normal(0, var, size=vector_dim)
    # b = np.random.normal(0, var, size=vector_dim)
    a = np.array([1.1, 2.02, 3.36, 4.87])
    b = np.array([4.5, 2.8, 3.51, 4.4])

    nested_inner_product = estimate_inner_product(a, b, nested_quantizer, method="nested")
    hierarchical_inner_product = estimate_inner_product(a, b, hierarchical_quantizer,
                                                        method="hierarchical", lookup_table=lookup_table)

    print(f"Nested Quantizer Inner Product: {nested_inner_product}")
    print(f"Hierarchical Quantizer Inner Product: {hierarchical_inner_product}")
    print(f"Actual Inner Product: {np.dot(a, b)}")

if __name__ == "__main__":
    main()