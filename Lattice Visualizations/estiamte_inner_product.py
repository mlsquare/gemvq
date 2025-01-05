import numpy as np
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_d3, calculate_mse
from closest_point import closest_point_Dn

def pad_vector(vec, lattice_dim):
    """Pad the vector with zeros to make its length a multiple of lattice_dim."""
    remainder = len(vec) % lattice_dim
    if remainder != 0:
        padding = lattice_dim - remainder
        vec = np.concatenate([vec, np.zeros(padding)])
    return vec

def precompute_inner_products(quantizer):
    """Precompute the inner product for all combinations of encoding vectors."""
    codebook = quantizer.create_codebook()
    lookup_table = {}

    for b1, x1 in codebook.items():
        for b2, x2 in codebook.items():
            inner_product = np.dot(x1, x2)
            lookup_table[(b1, b2)] = inner_product

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
            a_decoded = quantizer.decode(quantizer.encode(a_block))
            b_decoded = quantizer.decode(quantizer.encode(b_block))
            inner_product += np.dot(a_decoded, b_decoded)
        elif method == "hierarchical":
            a_enc = quantizer.encode(a_block)
            b_enc = quantizer.encode(b_block)
            inner_product += lookup_table[(tuple(a_enc), tuple(b_enc))]
        else:
            raise ValueError("Unsupported method. Use 'nested' or 'hierarchical'.")

    return inner_product

def main():
    q = 3
    vector_dim = 128
    G = get_d3()
    q_nn = closest_point_Dn
    var = 6

    nested_quantizer = NQuantizer(G, q_nn, beta=1, q=q)
    hierarchical_quantizer = HQuantizer(G, q_nn, beta=1, q=q)

    lookup_table = precompute_inner_products(hierarchical_quantizer)

    a = np.random.normal(0, var, size=vector_dim)
    b = np.random.normal(0, var, size=vector_dim)

    nested_inner_product = estimate_inner_product(a, b, nested_quantizer, method="nested")
    hierarchical_inner_product = estimate_inner_product(a, b, hierarchical_quantizer, method="hierarchical", lookup_table=lookup_table)

    print(f"Nested Quantizer Inner Product: {nested_inner_product}")
    print(f"Hierarchical Quantizer Inner Product: {hierarchical_inner_product}")

if __name__ == "__main__":
    main()